"""原有限宽 tanh 网络的联合输出 CGF、BCE Chernoff 代价及鞍点近似。

不读取 SMC 目标，不训练网络，不生成参数 SMC 粒子。
Lambda(t) = (sum t)^2/(2m) + m log E exp((t.h)^2/(2m)).
"""

from __future__ import annotations

import math
import time
from functools import lru_cache

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import brentq, minimize
from scipy.special import log_ndtr, xlogy
from numpy.polynomial.hermite import hermgauss
import torch


def truth_inputs(bits):
    return ((np.arange(2**bits)[:, None] >> np.arange(bits - 1, -1, -1)) & 1).astype(float)


def signs_for(function_id, bits):
    return 2.0 * ((int(function_id) >> np.arange(2**bits)) & 1) - 1.0


class FiniteCGF:
    def __init__(self, bits=3, width=16, order=24, device="cuda:1", activation="tanh"):
        if order % 2:
            raise ValueError("采用偶数阶求积，以准确利用整体反号对称性。")
        self.bits, self.width, self.order = bits, width, order
        self.device, self.activation = device, activation
        self.calls = 0
        nodes, weights = hermgauss(order)
        nodes, weights = nodes * math.sqrt(2), weights / math.sqrt(math.pi)
        axes = [np.arange(order // 2, order)] + [np.arange(order)] * bits
        # 整体反号使 h 变号，但 CGF 的被积函数不变，因此积分点减半。
        indices = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, bits + 1)
        latent = torch.tensor(nodes[indices], dtype=torch.float64, device=device)
        log_weights = np.log(weights[indices]).sum(1) + math.log(2)
        self.log_weights = torch.tensor(log_weights, dtype=torch.float64, device=device)
        self.log_weights -= torch.logsumexp(self.log_weights, 0)
        design = np.column_stack((truth_inputs(bits), np.ones(2**bits))) / math.sqrt(bits)
        design = torch.tensor(design, dtype=torch.float64, device=device)
        pre = latent @ design.T
        self.h = pre.tanh() if activation == "tanh" else pre
        self.h = self.h.contiguous()
        del latent, indices, pre
        self.node_count = len(self.h)
        self.r = 2**bits
        self._last_t, self._last_stats = None, None

    def evaluate(self, t, hessian=False):
        t = np.asarray(t, dtype=float)
        if not hessian and self._last_t is not None and np.array_equal(t, self._last_t):
            return self._last_stats
        self.calls += 1
        with torch.no_grad():
            v = torch.tensor(t, dtype=torch.float64, device=self.device)
            projection = self.h @ v
            score = self.log_weights + projection.square() / (2 * self.width)
            lse = torch.logsumexp(score, 0)
            p = torch.exp(score - lse)
            eh = self.h.T @ (p * projection)
            second = (p * projection.square()).sum()
            bias_cost = v.sum().square() / (2 * self.width)
            cgf = self.width * lse + bias_cost
            gradient = eh + v.sum() / self.width
            hidden_kl = second / 2 - self.width * lse
            readout_kl = second / 2 + bias_cost
            values = torch.cat((torch.stack((cgf, hidden_kl, readout_kl)), gradient)).cpu().numpy()
            result = {"cgf": float(values[0]), "hidden_kl": float(values[1]),
                      "readout_kl": float(values[2]), "mean": values[3:]}
            if hessian:
                gram = (self.h.T * p) @ self.h
                fourth = (self.h.T * (p * projection.square())) @ self.h
                cov = gram + (fourth - torch.outer(eh, eh)) / self.width
                cov += 1 / self.width
                result["covariance"] = cov.cpu().numpy()
                result["hidden_gram"] = gram.cpu().numpy()
        if not hessian:
            self._last_t, self._last_stats = t.copy(), result
        return result

    def initial(self, signed, epsilon):
        target_margin = -math.log(math.expm1(epsilon))
        def derivative(rho):
            return float(signed @ self.evaluate(rho * signed)["mean"] - self.r * target_margin)
        high = 1.0
        while derivative(high) < 0:
            high *= 2
            if high > 1e6:
                raise RuntimeError("标量初值未能包住驻点。")
        rho = brentq(derivative, 0, high, xtol=1e-10)
        return np.full(self.r + 1, rho)

    def solve(self, function_id, epsilon, initial=None):
        signed = signs_for(function_id, self.bits)
        p0 = -math.expm1(-epsilon)
        r = self.r
        start_time, calls0 = time.perf_counter(), self.calls
        x0 = self.initial(signed, epsilon) if initial is None else np.asarray(initial, dtype=float)

        def objective(x):
            w, alpha = x[:-1], x[-1] / p0
            if alpha <= 0 or np.any(w <= 0) or np.any(w >= alpha):
                violation = np.maximum(w - alpha + 1e-12, 0)
                grad = np.r_[2e6 * violation, -2e6 * violation.sum() / p0]
                return 1e6 + 1e6 * float(violation @ violation), grad
            p = w / alpha
            entropy = -xlogy(p, p) - xlogy(1 - p, 1 - p)
            stats = self.evaluate(signed * w)
            dual = alpha * (entropy.sum() - r * epsilon) - stats["cgf"]
            dw = np.log1p(-p) - np.log(p) - signed * stats["mean"]
            da = (-np.log1p(-p).sum() - r * epsilon) / p0
            return -float(dual), -np.r_[dw, da]

        constraint = np.column_stack((-p0 * np.eye(r), np.ones(r)))
        result = minimize(objective, x0, jac=True, method="SLSQP",
                          bounds=[(1e-10, None)] * (r + 1),
                          constraints={"type": "ineq", "fun": lambda x: constraint @ x - 1e-12,
                                       "jac": lambda x: constraint},
                          options={"maxiter": 1000, "ftol": 1e-11})
        x = result.x
        t, alpha = signed * x[:-1], x[-1] / p0
        stats = self.evaluate(t, hessian=True)
        z = stats["mean"]
        loss = float(np.logaddexp(0, -signed * z).mean())
        margins_dual = np.log(alpha / x[:-1] - 1)
        residual = float(np.max(np.abs(signed * z - margins_dual)))
        action = -float(result.fun)
        covariance = (stats["covariance"] + stats["covariance"].T) / 2
        chol = np.linalg.cholesky(covariance)
        normal = chol.T @ t
        normal_scale = float(np.linalg.norm(normal))
        tangent = null_space(normal[None, :])
        probs = 1 / (1 + np.exp(np.clip(signed * z, -700, 700)))
        loss_hessian = np.diag(probs * (1 - probs) / r)
        multiplier = r * alpha
        curvature = np.eye(r - 1) + multiplier * tangent.T @ chol.T @ loss_hessian @ chol @ tangent
        curvature_sign, curvature_logdet = np.linalg.slogdet(curvature)
        if curvature_sign <= 0:
            raise RuntimeError("切向修正非正定。")
        # 线性 Gaussian 半空间下此因子精确；非线性事件下仅作鞍点近似。
        log_boundary = normal_scale**2 / 2 + float(log_ndtr(-normal_scale))
        approx_nats = action - log_boundary + curvature_logdet / 2
        return {"bits": self.bits, "width": self.width, "order": self.order,
                "function_id": function_id, "epsilon": epsilon,
                "action_nats": action, "action_bits": action / math.log(2),
                "saddle_K_bits": approx_nats / math.log(2),
                "hidden_kl_nats": stats["hidden_kl"], "readout_kl_nats": stats["readout_kl"],
                "cgf": stats["cgf"], "t": t.tolist(), "dual_variables": x.tolist(),
                "mean_logits": z.tolist(), "mean_bce": loss,
                "constraint_relative_error": abs(loss / epsilon - 1),
                "kkt_margin_residual": residual, "normal_scale": normal_scale,
                "curvature_logdet": float(curvature_logdet),
                "tilted_covariance": covariance.tolist(), "hidden_gram": stats["hidden_gram"].tolist(),
                "duality_identity_error": abs(action - float(t @ z - stats["cgf"])),
                "kl_split_error": abs(float(t @ z - stats["cgf"]) - stats["hidden_kl"] - stats["readout_kl"]),
                "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
                "iterations": int(result.nit), "cgf_calls": self.calls - calls0,
                "seconds": time.perf_counter() - start_time,
                "numerically_converged": bool(abs(loss / epsilon - 1) < 2e-5 and residual < 2e-4)}
