# 正式运行器片段，由builder放在已验证的模型/前向定义后。

class Config:
    MODEL_SIZE = "compact"
    ARCHITECTURES = ("mlp", "rnn", "lstm", "transformer", "vit", "convnext",
                     "swin", "unet", "unet_global", "swin_unet")
    RULES = (204, 90)                    # 可追加30、110，已有项目不重跑
    POINTS = 16                         # 256才是全规则约束；此项改变需新目录
    IMAGE_SIZE = 32
    TRACKS = ("cells_mean_bce", "cells_max_bce", "native_mse")
    PARTICLES = 128                      # 与5090测速一致；加粒子须另设RUN_TAG
    REPLICAS = 4
    ADAPT_SWEEPS = 2
    MUTATION_SWEEPS = 4
    EVAL_PARTICLES = 128
    BCE_THRESHOLDS = (1., .8, .7, math.log(2), .6, .5, .4, .3, .2, .15,
                      .1, .07, .05, .03, .02, .01, .007, .005, .003, .002, .001, .0005, .0001)
    MSE_THRESHOLDS = (2., 1., .5, .3, .2, .1, .05, .03, .02, .01, .005,
                      .003, .002, .001, .0005, .0002, .0001, .00005, .00001)
    STOP_BCE = .001
    STOP_MSE = .0001
    SECONDS_PER_REPLICA = 180.
    MAX_RUN_HOURS = 8.
    MAX_LEVELS_PER_REPLICA_PER_RUN = 20000
    CHECKPOINT_SECONDS = 60.
    LOG_SECONDS = 30.
    PRIOR_SAMPLES = 2048                 # 真正对全部网络样本统计，不用单个网络的RMS冒充平均
    PRIOR_CUTOFFS_BCE = (1., .8, .7, .6, .5)
    PRIOR_CUTOFFS_MSE = (2., 1., .5, .2, .1)
    SNAPSHOT_WEIGHTS = 2
    SAVE_FULL_INPUT_LOGITS = True        # 全256输入的格点logits；只诊断，不影响目标
    FULL_PROBE_INPUT_BATCH = 16
    DATA_SEED = 202609120701
    SMC_SEED = 202609120702
    PRIOR_SEED = 202609120703
    DEVICE = "cuda:0"
    OUTPUT_ROOT = Path("/root/autodl-tmp") if os.name != "nt" else Path.cwd() / "tmp"
    RUN_TAG = "compact_curves_v1"
    PACKAGE_CHECKPOINTS = False
    MAKE_PLOTS = True
    SELF_TEST_ONLY = False
    # 只控制本次队列，不改变科学定义。例: SKIP_ARCHITECTURES=("unet",)。
    SKIP_ARCHITECTURES = ()
    REPLICA_IDS = None
    JOB_KEYS = None                      # 精确选择如("swin_unet/rule90/cells_max_bce/r0",)


PROTOCOL = "cross_architecture_bit_rule_curves_v1"
IMAGE_MODELS = {"unet", "unet_global", "swin_unet"}
STOP = False
LOG_PATH = None
RUN_STARTED = None


def replace_file(temporary, destination):
    for attempt in range(8):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.1 * (attempt + 1))


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def tracks_for(name):
    return tuple(t for t in Config.TRACKS if t != "native_mse" or name in IMAGE_MODELS)


def target_grid(track):
    return Config.BCE_THRESHOLDS if "bce" in track else Config.MSE_THRESHOLDS


def stop_at(track):
    return Config.STOP_BCE if "bce" in track else Config.STOP_MSE


def elapsed_budget_exhausted():
    return RUN_STARTED is not None and Config.MAX_RUN_HOURS > 0 and time.perf_counter() - RUN_STARTED >= Config.MAX_RUN_HOURS * 3600


def should_stop(out):
    return STOP or (out / "STOP").exists() or elapsed_budget_exhausted()


def request_stop(signum, frame):
    global STOP
    if STOP:
        raise KeyboardInterrupt
    STOP = True
    print("\n停止请求已收到，完成当前SMC层后保存并打包。再次中断会回退到上个完整检查点。", flush=True)


def science_signature():
    def encode(value):
        if isinstance(value, types.CodeType):
            return [value.co_code.hex(), value.co_names, value.co_varnames, value.co_argcount,
                    value.co_kwonlyargcount, [encode(v) for v in value.co_consts]]
        return repr(value)
    code = []
    for obj in (smc_level, initialize_state, generator_for, SamplingEvaluator):
        functions = [(obj.__name__, obj)] if not isinstance(obj, type) else vars(obj).items()
        for name, function in functions:
            function = getattr(function, "__wrapped__", function)
            if isinstance(function, types.FunctionType):
                code.append((name, encode(function.__code__)))
    names = ("MODEL_SIZE", "POINTS", "IMAGE_SIZE", "PARTICLES", "REPLICAS", "ADAPT_SWEEPS",
             "MUTATION_SWEEPS", "EVAL_PARTICLES", "DATA_SEED", "SMC_SEED", "PRIOR_SEED",
             "BCE_THRESHOLDS", "MSE_THRESHOLDS", "FULL_PROBE_INPUT_BATCH")
    return ready(dict(protocol=PROTOCOL, config={n: getattr(Config, n) for n in names},
                      model_source_sha256=BUNDLED_MODEL_SOURCE_SHA256,
                      sampler_sha256=hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest(),
                      torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                      numerical="float32 parameters/forward; Python float64 accumulated log mass; TF32/AMP off",
                      reference_measure="iid standard Gaussian neural coordinates; affine prior per parameter in manifest",
                      buffers="eval; BatchNorm running mean=0, variance=1 fixed; all learnable parameters sampled"))


def generator_for(device, key):
    offset = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    return torch.Generator(device=device).manual_seed((Config.SMC_SEED + offset) % (2**63 - 1))


def job_key(name, rule, track, replica):
    return f"{name}/rule{rule}/{track}/r{replica}"


class SamplingEvaluator(Evaluator):
    def raw_on(self, theta, x):
        return functional_call(self.model, (self.parameters(theta), self.buffers), (x,), strict=True)

    @torch.no_grad()
    def raw_batch(self, theta, x=None):
        x = self.x if x is None else x
        return torch.cat([vmap(lambda t: self.raw_on(t, x), randomness="error")(part)
                          for part in theta.split(Config.EVAL_PARTICLES)], 0)


def initialize_state(evaluator, generator):
    theta = torch.randn(Config.PARTICLES, evaluator.dimensions, device=evaluator.center.device, generator=generator)
    scores = evaluator.scores(theta)
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError("初始粒子有非有限loss")
    return dict(theta=theta, scores=scores, ancestry=torch.arange(Config.PARTICLES, device=theta.device),
                K=0., cutoff=float("inf"), node=0, level=0, scales=[.05, .08], points=[], levels=[],
                smc_seconds=0., diagnostic_seconds=0.)


@torch.no_grad()
def smc_level(evaluator, state, generator):
    target = target_grid(evaluator.track)[state["node"]]
    adaptive = float(torch.quantile(state["scores"], .5))
    effective_target = float(torch.tensor(target, dtype=state["scores"].dtype))
    cutoff = min(state["cutoff"], target if adaptive <= effective_target else adaptive)
    keep = torch.nonzero(state["scores"] <= cutoff).flatten()
    if not len(keep):
        raise RuntimeError("零幸存粒子，不伪造体积；保留上一检查点")
    if len(keep) < Config.PARTICLES:
        chosen = keep[torch.randint(len(keep), (Config.PARTICLES,), device=keep.device, generator=generator)]
        state["theta"] = state["theta"][chosen].clone()
        state["scores"] = state["scores"][chosen].clone()
        state["ancestry"] = state["ancestry"][chosen].clone()
    # 全存活时不凭空重采样，避免只增加重复粒子；质量因子仍为1。
    state["K"] -= math.log2(len(keep) / Config.PARTICLES)
    head = sum(p["count"] for p in evaluator.metadata[-2:])
    blocks = ((0, evaluator.dimensions), (evaluator.dimensions-head, evaluator.dimensions))
    rates, jumps = [0., 0.], [0., 0.]
    for sweep in range(Config.ADAPT_SWEEPS + Config.MUTATION_SWEEPS):
        for j, (lo, hi) in enumerate(blocks):
            rho = state["scales"][j]
            proposal = state["theta"].clone()
            old = state["theta"][:, lo:hi]
            proposal[:, lo:hi].mul_(math.sqrt(1 - rho*rho))
            noise = torch.randn(old.shape, device=old.device, generator=generator)
            proposal[:, lo:hi].add_(noise, alpha=rho)
            del noise
            values = evaluator.scores(proposal)
            accept = torch.isfinite(values) & (values <= cutoff)
            rate = float(accept.float().mean())
            probe = min(8, len(old))
            jump = float(((proposal[:probe, lo:hi]-old[:probe]).square().mean(1)*accept[:probe]).mean())
            state["theta"][accept] = proposal[accept]
            state["scores"][accept] = values[accept]
            if sweep < Config.ADAPT_SWEEPS:
                state["scales"][j] = min(.8, max(1e-5, rho * math.exp(.35*(rate-.3))))
            else:
                rates[j] += rate / Config.MUTATION_SWEEPS
                jumps[j] += jump / Config.MUTATION_SWEEPS
            del proposal, old, values, accept
    state["level"] += 1
    state["cutoff"] = cutoff
    hit = cutoff == target
    state["node"] += int(hit)
    if not bool(torch.isfinite(state["scores"]).all()) or bool((state["scores"] > cutoff).any()):
        raise RuntimeError("采样结束后粒子违反约束或出现非有限值")
    effective = float(torch.tensor(cutoff, dtype=state["scores"].dtype))
    warning = []
    if min(rates) < .03:
        warning.append("LOW_ACCEPTANCE")
    if min(state["scales"]) <= 1.01e-5:
        warning.append("PROPOSAL_SCALE_FLOOR")
    return dict(level=state["level"], cutoff=cutoff, effective_float32_cutoff=effective,
                K_bits=state["K"], survival=len(keep)/Config.PARTICLES, acceptance=rates,
                coordinate_esjd_probe=jumps, scales=list(state["scales"]),
                original_ancestors=int(state["ancestry"].unique().numel()), warnings=warning), hit


def checkpoint(path, state, generator, signature, elapsed):
    temporary = path.with_name(path.name + ".tmp")
    cpu = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else ready(v) if isinstance(v, (dict, list)) else v
           for k, v in state.items()}
    torch.save(dict(state=cpu, rng=generator.get_state().cpu(), signature=signature, elapsed_s=float(elapsed)), temporary)
    replace_file(temporary, path)


def restore(path, generator, device, signature):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["signature"] != signature:
        raise ValueError(f"{path}的模型/数据/采样配置或软件版本不同，拒绝混用")
    generator.set_state(saved["rng"])
    state = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in saved["state"].items()}
    return state, float(saved["elapsed_s"])


def full_data(rule, device):
    data = dataset(256, rule, device)
    order = np.argsort(data["input_ids"])
    indices = torch.tensor(order, device=device)
    return {k: v[indices] if isinstance(v, torch.Tensor) else v[order] for k, v in data.items()}


def output_statistics(z, labels, decision_threshold=0.):
    z = np.asarray(z, dtype=np.float64)
    labels = np.asarray(labels)
    margins = z * (2*labels[None]-1)
    per_bit_loss = np.logaddexp(0, -margins)
    predicted = z >= decision_threshold
    correct = predicted == labels[None].astype(bool)
    vote = predicted.mean(0)
    return dict(decision_threshold=decision_threshold,
                bit_accuracy=float(correct.mean()), full_mapping_fraction=float(correct.all(axis=(1, 2)).mean()),
                agreement_majority=float(np.maximum(vote, 1-vote).mean()),
                agreement_pair_with_replacement=float((vote*vote+(1-vote)**2).mean()),
                mean_bce=float(per_bit_loss.mean()),
                margin_min=float(margins.min()), margin_mean=float(margins.mean()),
                mean_within_network_margin_std=float(margins.reshape(len(z), -1).std(1).mean()),
                margin_quantiles=np.quantile(margins, [.01, .1, .5, .9, .99]).tolist(),
                mean_network_max_bit_bce=float(per_bit_loss.reshape(len(z), -1).max(1).mean()),
                mean_network_min_bit_bce=float(per_bit_loss.reshape(len(z), -1).min(1).mean()))


@torch.no_grad()
def record_node(directory, evaluator, state, rule):
    raw = evaluator.raw_batch(state["theta"])
    train_z = vmap(cell_outputs)(raw).cpu().numpy()
    y = evaluator.data["labels"].cpu().numpy()
    threshold = .5 if evaluator.track == "native_mse" and evaluator.name == "unet" else 0.
    stats = output_statistics(train_z, y, threshold)
    p = len(train_z)
    arrays = dict(train_cell_logits=train_z, train_labels=y, scores=state["scores"].cpu().numpy(),
                  train_input_ids=evaluator.data["input_ids"],
                  theta=state["theta"][:min(Config.SNAPSHOT_WEIGHTS, p)].cpu().numpy(),
                  raw_outputs=raw[:min(Config.SNAPSHOT_WEIGHTS, p)].cpu().numpy())
    if evaluator.name in IMAGE_MODELS:
        prediction = raw if evaluator.name == "unet" else raw.sigmoid()
        per_particle_mse = (prediction-evaluator.data["targets"][None]).square().flatten(1).mean(1)
        arrays["native_mse_by_particle"] = per_particle_mse.cpu().numpy()
        stats["native_train_mse"] = float(per_particle_mse.mean())
        stats["native_pixel_accuracy"] = float(((prediction >= .5) == evaluator.data["targets"][None].bool()).float().mean())
        del prediction, per_particle_mse
    if "bce" in evaluator.track:
        bound = math.log(2) / (len(y)*8) if evaluator.track == "cells_mean_bce" else math.log(2)
        if state["cutoff"] < bound and stats["full_mapping_fraction"] < 1:
            raise RuntimeError("BCE已低于硬拟合保证阈值，但观测格点仍有错误")
    del raw
    if Config.SAVE_FULL_INPUT_LOGITS:
        probe = full_data(rule, evaluator.center.device)
        x = model_input(evaluator.name, probe)
        logits = []
        for start in range(0, 256, Config.FULL_PROBE_INPUT_BATCH):
            raw = evaluator.raw_batch(state["theta"], x[start:start+Config.FULL_PROBE_INPUT_BATCH])
            logits.append(vmap(cell_outputs)(raw).cpu().numpy())
            del raw
        all_z = np.concatenate(logits, axis=1)
        all_y = probe["labels"].cpu().numpy()
        packed = np.packbits((all_z >= threshold).reshape(p, -1), axis=1, bitorder="little")
        _, counts = np.unique(packed, axis=0, return_counts=True)
        weights = counts / p
        stats["full_input_space"] = output_statistics(all_z, all_y, threshold)
        stats["full_input_space"]["empirical_function_count"] = len(counts)
        stats["full_input_space"]["empirical_function_entropy_bits"] = float(-(weights*np.log2(weights)).sum())
        heldout = np.ones(256, dtype=bool)
        heldout[evaluator.data["input_ids"]] = False
        stats["held_out"] = output_statistics(all_z[:, heldout], all_y[heldout], threshold) if heldout.any() else None
        arrays.update(full_cell_logits=all_z, full_labels=all_y, full_input_ids=np.arange(256),
                      hard_functions_packed=packed, held_out_mask=heldout)
    filename = f"node_{state['node']:03d}.npz"
    atomic_npz(directory / filename, **arrays)
    return dict(output_statistics=stats, snapshot=filename, snapshot_sha256=sha_file(directory / filename),
                empirical_particles=p, note="SMC粒子有相关性；经验agreement不是独立SGD种子的共识")


def source_archive(out):
    source, origin = None, None
    filename = globals().get("__file__")
    if filename and Path(filename).is_file():
        source, origin = Path(filename).read_text(encoding="utf-8"), "python_file"
    else:
        getter = globals().get("get_ipython")
        if callable(getter):
            try:
                history = getter().history_manager.input_hist_raw
                source = next((cell for cell in reversed(history) if PROTOCOL in cell and "class Config:" in cell), None)
                origin = "jupyter_history"
            except (AttributeError, IndexError):
                pass
    if source:
        h = hashlib.sha256(source.encode()).hexdigest()
        folder = out / "sources"
        folder.mkdir(exist_ok=True)
        atomic_text(folder / f"experiment_{h[:16]}.py", source)
        return dict(saved=True, sha256=h, origin=origin)
    return dict(saved=False, note="没有__file__且未取得Jupyter历史；保留模型源码/采样代码签名，不因此报错")


def write_replica_artifacts(directory, state, signature, elapsed, status, error=None):
    atomic_json(directory / "points.json", state["points"])
    atomic_json(directory / "levels.json", state["levels"])
    last = state["levels"][-1] if state["levels"] else {}
    summary = dict(signature=signature, status=status, error=error, cutoff=state["cutoff"],
                   K_bits=state["K"], levels=state["level"], fixed_nodes=len(state["points"]),
                   elapsed_s=elapsed, smc_seconds=state["smc_seconds"], diagnostic_seconds=state["diagnostic_seconds"],
                   acceptance=last.get("acceptance"), warnings=last.get("warnings", []),
                   reliability="FINITE_LOSS_ESTIMATE_NOT_CONVERGENCE_CERTIFIED")
    atomic_json(directory / "summary.json", summary)
    return ready(summary)


def run_replica(out, evaluator, rule, replica, signature):
    key = job_key(evaluator.name, rule, evaluator.track, replica)
    directory = out / "jobs" / key
    directory.mkdir(parents=True, exist_ok=True)
    local_signature = ready(dict(science=signature, job=key, prior=evaluator.metadata,
                                data_sha256=digest(evaluator.x.cpu().numpy()),
                                labels_sha256=digest(evaluator.data["labels"].cpu().numpy())))
    summary_path, check_path = directory / "summary.json", directory / "checkpoint.pt"
    if summary_path.exists():
        old = read_json(summary_path)
        if old["signature"] != local_signature:
            raise ValueError(f"{key}结果配置不同，拒绝混合")
        if old["cutoff"] is not None and old["cutoff"] <= stop_at(evaluator.track):
            log(f"SKIP {key} 已达目标")
            return "skipped"
        if not check_path.exists():
            raise ValueError(f"{key}已有结果但缺少粒子检查点，不能假装续跑")
    generator = generator_for(evaluator.center.device, key)
    if check_path.exists():
        state, base = restore(check_path, generator, evaluator.center.device, local_signature)
    else:
        state, base = initialize_state(evaluator, generator), 0.
        checkpoint(check_path, state, generator, local_signature, base)
    start = last_save = last_log = time.perf_counter()
    first_level = state["level"]
    status, error, inside = "time_budget", None, False
    log(f"START {key} | level={state['level']} | loss<={state['cutoff']:.6g} | {Config.PARTICLES}粒子")
    try:
        while state["cutoff"] > stop_at(evaluator.track):
            now = time.perf_counter()
            if should_stop(out):
                status = "paused"
                break
            if now-start >= Config.SECONDS_PER_REPLICA:
                break
            if state["level"]-first_level >= Config.MAX_LEVELS_PER_REPLICA_PER_RUN:
                status = "level_budget"
                break
            inside = True
            begin_level = time.perf_counter()
            row, hit = smc_level(evaluator, state, generator)
            state["smc_seconds"] += time.perf_counter()-begin_level
            row["elapsed_s"] = base + time.perf_counter()-start
            state["levels"].append(row)
            if hit:
                begin_diagnostics = time.perf_counter()
                observation = record_node(directory, evaluator, state, rule)
                state["diagnostic_seconds"] += time.perf_counter()-begin_diagnostics
                point = dict(row, **observation)
                state["points"].append(point)
                stats = observation["output_statistics"]
                held = stats.get("held_out")
                held_text = f" heldout位正确率={held['bit_accuracy']:.2%}" if held else ""
                log(f"HIT {key} loss<={row['cutoff']:.6g} | K={row['K_bits']:.2f}bit | "
                    f"约束格点全对粒子={stats['full_mapping_fraction']:.1%}{held_text}")
            inside = False
            now = time.perf_counter()
            if now-last_save >= Config.CHECKPOINT_SECONDS:
                elapsed = base+now-start
                checkpoint(check_path, state, generator, local_signature, elapsed)
                write_replica_artifacts(directory, state, local_signature, elapsed, "running")
                last_save = time.perf_counter()
            if now-last_log >= Config.LOG_SECONDS:
                log(f"{key} level={state['level']} loss<={state['cutoff']:.6g} | "
                    f"K={state['K']:.1f} | 接受率={row['acceptance']} | 本次{now-start:.0f}s")
                last_log = now
        if state["cutoff"] <= stop_at(evaluator.track):
            status = "completed"
    except KeyboardInterrupt:
        global STOP
        STOP = True
        status = "paused"
        if inside:
            state, base = restore(check_path, generator, evaluator.center.device, local_signature)
            start = time.perf_counter()
            log(f"{key}半层强制中断，回到最后完整检查点")
    except Exception as exc:
        error, status = repr(exc), "error"
        # 释放失败提案的traceback引用后再恢复，避免OOM时仍占用整群提案显存。
        exc.__traceback__ = None
        state = None
        torch.cuda.empty_cache()
        state, base = restore(check_path, generator, evaluator.center.device, local_signature)
        start = time.perf_counter()
        log(f"ERROR {key}: {error}，保留最后完整状态，不自动复测")
    elapsed = base+time.perf_counter()-start
    checkpoint(check_path, state, generator, local_signature, elapsed)
    result = write_replica_artifacts(directory, state, local_signature, elapsed, status, error)
    log(f"END {key}: {status} | loss<={state['cutoff']:.6g} | 累计{elapsed/60:.1f}分钟")
    return result["status"]


@torch.no_grad()
def preflight(out, name, model, data_by_rule, signature):
    folder = out / "models" / name
    folder.mkdir(parents=True, exist_ok=True)
    stored = folder / "preflight.json"
    if stored.exists():
        result = read_json(stored)
        if result["signature"] != signature:
            raise ValueError("预检配置不一致")
        existing_pairs = {(r["rule"], r["track"]) for r in result["prior_calibration"]}
        requested_pairs = {(r, t) for r in Config.RULES for t in tracks_for(name)}
        if result["prior_samples"] >= Config.PRIOR_SAMPLES and requested_pairs <= existing_pairs:
            return result
    rule = Config.RULES[0]
    evaluator = SamplingEvaluator(name, model, data_by_rule[rule], "cells_mean_bce")
    g = torch.Generator(device=evaluator.center.device).manual_seed(Config.PRIOR_SEED)
    theta = torch.randn(2, evaluator.dimensions, device=evaluator.center.device, generator=g)
    batched = evaluator.scores(theta)
    serial = torch.stack([evaluator.single(t) for t in theta])
    if not torch.allclose(batched, serial, atol=2e-6, rtol=2e-4):
        raise RuntimeError("批量与逐模型前向不一致")
    actual = evaluator.raw(theta[0])
    model.load_state_dict(dict(evaluator.parameters(theta[0]), **evaluator.buffers), strict=True)
    if not torch.allclose(actual, model(evaluator.x), atol=2e-6, rtol=2e-5):
        raise RuntimeError("神经坐标与直接参数装载不一致")
    atomic_json(folder / "parameter_prior.json", evaluator.metadata)
    atomic_text(folder / "architecture.txt", str(model))
    atomic_npz(folder / "forward_check.npz", theta=theta.cpu().numpy(), raw_outputs=evaluator.raw_batch(theta).cpu().numpy())
    comparisons = {(r, t): [] for r in Config.RULES for t in tracks_for(name)}
    rms = []
    for start in range(0, Config.PRIOR_SAMPLES, Config.EVAL_PARTICLES):
        count = min(Config.EVAL_PARTICLES, Config.PRIOR_SAMPLES-start)
        prior = torch.randn(count, evaluator.dimensions, device=evaluator.center.device, generator=g)
        raw = evaluator.raw_batch(prior)
        rms.extend(raw.square().flatten(1).mean(1).sqrt().cpu().tolist())
        for (r, track), values in comparisons.items():
            values.extend(vmap(lambda z: score_raw(z, name, track, data_by_rule[r]))(raw).cpu().tolist())
        del raw, prior
        if should_stop(out):
            break
    calibration = []
    for (r, track), values in comparisons.items():
        array = np.asarray(values)
        thresholds = Config.PRIOR_CUTOFFS_BCE if "bce" in track else Config.PRIOR_CUTOFFS_MSE
        for cutoff in thresholds:
            count = int((array <= np.float32(cutoff)).sum())
            calibration.append(dict(rule=r, track=track, cutoff=cutoff, hits=count,
                                    samples=len(values), probability=count/len(values)))
    result = dict(signature=signature, architecture=name, parameters=evaluator.dimensions,
                  prior_samples=len(rms), prior_rms_mean=float(np.mean(rms)), prior_rms_sd=float(np.std(rms)),
                  vmap_serial_error=float((batched-serial).abs().max()), prior_calibration=calibration,
                  reference="参考测度按parameter_prior.json，不自动缩放输出或借用SGD落点")
    atomic_json(stored, result)
    return ready(result)


def summarize(out, signature):
    groups, coverage, errors = {}, [], []
    failure_path = out / "preflight_errors.json"
    if failure_path.exists():
        errors.extend(dict(job=e.get("job", e.get("architecture", "unknown")+"/preflight"), error=e["error"])
                      for e in read_json(failure_path))
    for path in sorted(out.glob("jobs/*/rule*/*/r*/summary.json")):
        info = read_json(path)
        if info["signature"]["science"] != signature:
            raise ValueError("汇总遇到不同科学配置")
        key = info["signature"]["job"]
        name, rule, track, replica = key.split("/")
        info["completed_for_current_target"] = info["cutoff"] is not None and info["cutoff"] <= stop_at(track)
        coverage.append(dict(job=key, status=info["status"], cutoff=info["cutoff"], K_bits=info["K_bits"],
                             levels=info["levels"], fixed_nodes=info["fixed_nodes"], elapsed_s=info["elapsed_s"],
                             completed=info["completed_for_current_target"], error=info["error"]))
        if info["error"]:
            errors.append(dict(job=key, error=info["error"]))
        points_path = path.parent / "points.json"
        rows = read_json(points_path) if points_path.exists() else []
        groups.setdefault((name, rule, track), {})[replica] = {r["cutoff"]: r for r in rows}
    points, slopes = [], []
    for (name, rule, track), replicas in groups.items():
        grid = [v for v in target_grid(track) if any(v in r for r in replicas.values())]
        for cutoff in grid:
            members = {rep: rows[cutoff] for rep, rows in replicas.items() if cutoff in rows}
            values = np.array([r["K_bits"] for r in members.values()])
            warnings = sorted({w for r in members.values() for w in r["warnings"]})
            if len(values) < Config.REPLICAS:
                warnings.append("INCOMPLETE_REPLICAS")
            sd = float(values.std(ddof=1)) if len(values) > 1 else None
            if sd is not None and sd > max(5., .03*abs(np.median(values))):
                warnings.append("REPLICA_K_SPREAD")
            row = dict(architecture=name, rule=rule, track=track, cutoff=cutoff,
                       replicas=len(values), K_by_replica={k: r["K_bits"] for k, r in members.items()},
                       K_median_bits=float(np.median(values)), K_sd_bits=sd, warnings=warnings)
            stats = [r["output_statistics"] for r in members.values()]
            row["train_full_fit_fraction"] = float(np.mean([s["full_mapping_fraction"] for s in stats]))
            held = [s.get("held_out") for s in stats if s.get("held_out") is not None]
            row["held_out_bit_accuracy"] = float(np.mean([s["bit_accuracy"] for s in held])) if held else None
            points.append(row)
        # 使用预设相邻节点；不会跳过缺失节点后伪装成相同窗口。
        for high, low in zip(target_grid(track)[:-1], target_grid(track)[1:]):
            delta = {rep: rows[low]["K_bits"]-rows[high]["K_bits"] for rep, rows in replicas.items()
                     if high in rows and low in rows}
            if not delta:
                continue
            coefficients = np.array(list(delta.values())) / math.log2(high/low)
            median = float(np.median(coefficients))
            sd = float(coefficients.std(ddof=1)) if len(coefficients) > 1 else None
            warning = [] if len(delta) == Config.REPLICAS else ["INCOMPLETE_REPLICAS"]
            if sd is not None and sd > max(2., .15*abs(median)):
                warning.append("SLOPE_REPLICA_SPREAD")
            item = dict(architecture=name, rule=rule, track=track, high=high, low=low, replicas=len(delta),
                        c_log2_inverse_loss_median=median, c_sd=sd, delta_K_by_replica=delta, warnings=warning)
            if "bce" in track:
                a0, a1 = -math.log(math.expm1(high)), -math.log(math.expm1(low))
                item.update(a_low=a0, a_high=a1, c_a_median=float(np.median(list(delta.values()))/(a1-a0)),
                            exact_minimum_margin=track == "cells_max_bce")
                if a0 >= 0:
                    item["c_a_squared_median"] = float(np.median(list(delta.values()))/(a1*a1-a0*a0))
            slopes.append(item)
    planned = sum(len(tracks_for(n)) for n in Config.ARCHITECTURES) * len(Config.RULES) * Config.REPLICAS
    calibration_rows = []
    for (name, rule, track), replicas in groups.items():
        prior_path = out / "models" / name / "preflight.json"
        if not prior_path.exists():
            continue
        for ref in read_json(prior_path)["prior_calibration"]:
            if ref["rule"] != int(rule[4:]) or ref["track"] != track:
                continue
            values = [2**(-r[ref["cutoff"]]["K_bits"]) for r in replicas.values() if ref["cutoff"] in r]
            if not values:
                continue
            p0 = ref["probability"]
            reference_se = math.sqrt(p0*(1-p0)/ref["samples"])
            empirical_se = float(np.std(values, ddof=1)/math.sqrt(len(values))) if len(values) > 1 else 0.
            floor = p0*(1-p0)/(Config.PARTICLES*len(values))
            tolerance = 5*math.sqrt(reference_se**2 + max(empirical_se**2, floor)) + 3/ref["samples"]
            checkable = len(values) >= 3 and ref["hits"] >= 20 and ref["samples"]-ref["hits"] >= 20
            calibration_rows.append(dict(architecture=name, rule=rule, track=track, cutoff=ref["cutoff"],
                prior_samples=ref["samples"], prior_hits=ref["hits"], prior_probability=p0,
                smc_replicas=len(values), smc_probability_mean=float(np.mean(values)),
                heuristic_tolerance=tolerance, enough_samples_for_warning=checkable,
                warning=bool(checkable and abs(np.mean(values)-p0) > tolerance)))
    result = dict(planned_replicas=planned, recorded_replicas=len(coverage),
                  completed_replicas=sum(r["completed"] for r in coverage), errors=errors,
                  points=points, slopes=slopes, coverage=coverage, prior_calibration=calibration_rows,
                  caveat="只汇总已记录节点；误差条是副本分歧，不是收敛证明。格点BCE与整图MSE分开解释。")
    atomic_json(out / "analysis.json", result)
    write_csv(out / "coverage.csv", coverage)
    write_csv(out / "points.csv", points)
    write_csv(out / "slopes.csv", slopes)
    write_csv(out / "prior_calibration.csv", calibration_rows)
    return ready(result)


def make_plots(out, analysis):
    if not Config.MAKE_PLOTS:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        log("没有matplotlib，跳过曲线图；JSON/CSV/NPZ和采样均保留")
        return
    folder = out / "plots"
    folder.mkdir(exist_ok=True)
    for track in Config.TRACKS:
        names = [n for n in Config.ARCHITECTURES if track in tracks_for(n)]
        if not names:
            continue
        cols, rows = 3, math.ceil(len(names)/3)
        fig, axes = plt.subplots(rows, cols, figsize=(14, 3.7*rows), squeeze=False)
        for ax, name in zip(axes.flat, names):
            for rule in Config.RULES:
                data = [r for r in analysis["points"] if r["architecture"] == name and r["track"] == track and r["rule"] == f"rule{rule}"]
                data.sort(key=lambda r: r["cutoff"], reverse=True)
                if data:
                    x = [math.log2(1/r["cutoff"]) for r in data]
                    ax.errorbar(x, [r["K_median_bits"] for r in data],
                                yerr=[r["K_sd_bits"] or 0 for r in data], marker=".", capsize=2, label=f"rule {rule}")
                    for a, r in zip(x, data):
                        if r["warnings"]:
                            ax.scatter([a], [r["K_median_bits"]], marker="x", color="red", s=30, zorder=5)
            ax.set_title(name)
            ax.set_xlabel("log2(1 / loss limit)")
            ax.set_ylabel("K (bits)")
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.tick_params(labelsize=8)
            ax.grid(alpha=.25)
            if ax.lines:
                ax.legend()
        for ax in list(axes.flat)[len(names):]:
            ax.set_visible(False)
        fig.suptitle(track + " | replica SD; x = warning/incomplete")
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(folder / f"curves_{track}.png", dpi=150)
        plt.close(fig)
    for name in Config.ARCHITECTURES:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for rule in Config.RULES:
            for track in ("cells_mean_bce", "cells_max_bce"):
                data = [r for r in analysis["slopes"] if r["architecture"] == name and r["rule"] == f"rule{rule}" and r["track"] == track]
                data.sort(key=lambda r: r["low"], reverse=True)
                if data:
                    ax.plot([math.log2(1/math.sqrt(r["high"]*r["low"])) for r in data],
                            [r["c_log2_inverse_loss_median"] for r in data], ".-", label=f"{rule}: {track}")
        if ax.lines:
            ax.legend(fontsize=8)
        ax.set(xlabel="log2(1 / geometric-mean loss)", ylabel="delta K / delta log2(1/loss)", title=name)
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(folder / f"slopes_{name}.png", dpi=150)
        plt.close(fig)


def save_task_images(out, data_by_rule):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    size, scale = Config.IMAGE_SIZE, 4
    rules = list(data_by_rule)
    n = min(8, Config.POINTS)
    canvas = Image.new("RGB", ((len(rules)+1)*size*scale, n*(size*scale+20)), "white")
    draw = ImageDraw.Draw(canvas)
    inputs = data_by_rule[rules[0]]
    for i in range(n):
        cells = [inputs["images"][i, 0]] + [data_by_rule[r]["targets"][i, 0] for r in rules]
        for col, value in enumerate(cells):
            image = Image.fromarray((value.cpu().numpy()*255).astype(np.uint8)).convert("RGB")
            image = image.resize((size*scale, size*scale), resample=Image.Resampling.NEAREST)
            x, y = col*size*scale, i*(size*scale+20)
            canvas.paste(image, (x, y+20))
            text = f"x={int(inputs['input_ids'][i]):03d}" if col == 0 else f"rule {rules[col-1]}"
            draw.text((x+3, y+3), text, fill="black")
    canvas.save(out / "task_examples.png")


def package(out):
    destination = out.with_name(out.name + "_package.zip")
    temporary = destination.with_name(destination.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in sorted(out.rglob("*")):
            if not path.is_file() or path.suffix == ".tmp" or path.name in ("RUNNING.lock", "STOP"):
                continue
            if path.suffix == ".pt" and not Config.PACKAGE_CHECKPOINTS:
                continue
            archive.write(path, str(path.relative_to(out.parent)))
    replace_file(temporary, destination)
    return destination


def write_run_report(out, analysis, status):
    lines = ["# 跨架构bit规则体积曲线", "", f"状态：{status}。",
             f"已达当前目标的副本：{analysis['completed_replicas']}/{analysis['planned_replicas']}。", "",
             "默认16个固定输入约束、全部256输入只作诊断；不是把测试标签用于筛选。",
             "cells_mean_bce是格点平均BCE；cells_max_bce精确对应格点最小有符号margin下限，非等幅切片。",
             "native_mse是整张图像的实际MSE：普通U-Net直接raw输出，其余两种图像模型先sigmoid。",
             "普通U-Net的native-MSE按raw输出0.5解码，其它格点BCE/sigmoid-MSE按raw输出0解码。",
             "这些不是同维度的约束，不能把整图MSE与8格点BCE的K绝对值直接等同。", "",
             "warnings是副本分歧/采样诊断，不会触发全体自动recheck；无报警也不等于已证明收敛。",
             "prior_calibration.csv将浅层SMC与直接先验计数比较，容差是启发式诊断，不是深尾收敛检验。",
             "日志与level数反映进度；只有固定节点进入主曲线。不对未完成节点补值。",
             "K的误差条是副本SD；斜率先在同一副本做差，再对副本取中位数。", "",
             "每个node_*.npz保留训练格点logits、全部输入格点logits（如开启）、少量权重和原始输出。",
             "只读取points.json引用的快照；强制中断后可能存在尚未提交的旧尾部快照文件。",
             "checkpoint.pt保留在服务器原目录，用于续跑/续深；默认下载zip不包含它。", "",
             "本程序测静态参考测度，不是SGD训练；函数共识统计来自相关SMC粒子，不是独立训练种子。"]
    if analysis["errors"]:
        lines += ["", "## 错误", ""] + [f"- {e['job']}: {e['error']}" for e in analysis["errors"]]
    atomic_text(out / "report_zh.md", "\n".join(lines)+"\n")


def validate_config():
    if Config.MODEL_SIZE != "compact" or Config.IMAGE_SIZE < 32 or Config.IMAGE_SIZE % 32:
        raise ValueError("正式版默认compact，图像边长至少32且为32的倍数")
    if not 1 <= Config.POINTS <= 256 or Config.PARTICLES < 2 or Config.REPLICAS < 2:
        raise ValueError("POINTS/粒子数/副本数设置无效")
    if len(set(Config.ARCHITECTURES)) != len(Config.ARCHITECTURES) or len(set(Config.RULES)) != len(Config.RULES):
        raise ValueError("架构/规则不得重复")
    if not Config.ARCHITECTURES or not Config.RULES or any(not isinstance(r, int) or not 0 <= r <= 255 for r in Config.RULES):
        raise ValueError("需指定有效的架构和0至255的ECA规则")
    allowed = {"mlp", "rnn", "lstm", "transformer", "vit", "convnext", "swin", "unet", "unet_global", "swin_unet"}
    if not set(Config.ARCHITECTURES) <= allowed or not set(Config.SKIP_ARCHITECTURES) <= allowed:
        raise ValueError("未知架构名；diffusion不在本轮范围")
    if not Config.TRACKS or any(t not in ("cells_mean_bce", "cells_max_bce", "native_mse") for t in Config.TRACKS):
        raise ValueError("未知观测约束")
    if len(set(Config.TRACKS)) != len(Config.TRACKS):
        raise ValueError("观测约束不得重复")
    if Config.REPLICA_IDS is not None and (len(set(Config.REPLICA_IDS)) != len(Config.REPLICA_IDS)
            or any(not 0 <= r < Config.REPLICAS for r in Config.REPLICA_IDS)):
        raise ValueError("REPLICA_IDS必须是副本编号的无重复子集")
    for nodes, stop in ((Config.BCE_THRESHOLDS, Config.STOP_BCE), (Config.MSE_THRESHOLDS, Config.STOP_MSE)):
        if stop not in nodes or any(v <= 0 for v in nodes) or any(a <= b for a, b in zip(nodes[:-1], nodes[1:])):
            raise ValueError("loss网格必须严格递减且包含停止目标")
    if Config.MUTATION_SWEEPS < 1 or Config.ADAPT_SWEEPS < 0 or Config.EVAL_PARTICLES < 1 or Config.PRIOR_SAMPLES < 2:
        raise ValueError("采样/预检预算无效")
    if Config.SECONDS_PER_REPLICA <= 0 or Config.FULL_PROBE_INPUT_BATCH < 1:
        raise ValueError("运行时限/观测批次无效")


def main():
    global STOP, LOG_PATH, RUN_STARTED
    STOP = False
    validate_config()
    if not torch.cuda.is_available() or torch.device(Config.DEVICE).type != "cuda":
        raise RuntimeError("需要CUDA版torch，不自动退回CPU长时间采样")
    device = torch.device(Config.DEVICE)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    out = Config.OUTPUT_ROOT / f"results_cross_architecture_rule_volume_{Config.RUN_TAG}"
    out.mkdir(parents=True, exist_ok=True)
    signature = science_signature()
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and read_json(protocol_path)["signature"] != signature:
        raise ValueError("结果目录的模型/数据/采样配置或软件版本不一致；请更改RUN_TAG，不覆盖旧结果")
    lock = out / "RUNNING.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        raise RuntimeError(f"{lock}存在。确认旧进程已结束后移除此锁文件，不要同时写同一目录。") from None
    handlers, models, preflight_errors = {}, {}, []
    LOG_PATH = out / "run.log"
    RUN_STARTED = time.perf_counter()
    status = "running"
    try:
        atomic_json(lock, dict(pid=os.getpid(), host=platform.node(), started=time.time()))
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, request_stop)
        atomic_json(protocol_path, dict(signature=signature,
                    scope="fixed-D静态体积；全部输入探针只观察，不改变采样目标"))
        source = source_archive(out)
        atomic_json(out / f"invocation_{time.time_ns()}.json", dict(
            python=platform.python_version(), torch=str(torch.__version__), torchvision=str(torchvision.__version__),
            gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda, device=str(device), source=source,
            configuration={k: v for k, v in vars(Config).items() if k.isupper()}))
        data_by_rule = {rule: dataset(Config.POINTS, rule, device) for rule in Config.RULES}
        for rule, data in data_by_rule.items():
            atomic_npz(out / f"training_data_rule{rule}.npz", **{k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in data.items()})
            probe = full_data(rule, device)
            atomic_npz(out / f"full_truth_table_rule{rule}.npz", inputs=probe["bits"].cpu().numpy(),
                       labels=probe["labels"].cpu().numpy(), input_ids=probe["input_ids"])
        save_task_images(out, data_by_rule)
        log(f"=== 正式跨架构规则曲线 | {torch.cuda.get_device_name(device)} ===")
        log(f"{len(Config.ARCHITECTURES)}架构，规则{Config.RULES}，{Config.POINTS}/256输入约束，"
            f"{Config.REPLICAS}独立副本x{Config.PARTICLES}粒子")
        log(f"每副本本次最多{Config.SECONDS_PER_REPLICA:g}s；整次最多{Config.MAX_RUN_HOURS:g}h；"
            f"目标BCE={Config.STOP_BCE:g}、MSE={Config.STOP_MSE:g}；无自动recheck")
        for name in Config.ARCHITECTURES:
            if should_stop(out):
                break
            if name in Config.SKIP_ARCHITECTURES:
                continue
            try:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(Config.PRIOR_SEED)
                    model = make_model(name).to(device).eval()
                result = preflight(out, name, model, data_by_rule, signature)
                models[name] = model
                log(f"READY {name} {result['parameters']:,}参数 | 先验核对{result['prior_samples']}个网络")
            except (RuntimeError, ImportError, NotImplementedError) as exc:
                preflight_errors.append(dict(architecture=name, error=repr(exc)))
                log(f"PRECHECK ERROR {name}: {exc}；不换架构或降低精度")
        atomic_json(out / "preflight_errors.json", preflight_errors)
        replicas = range(Config.REPLICAS) if Config.REPLICA_IDS is None else Config.REPLICA_IDS
        jobs = [(name, rule, track, rep) for rep in replicas for rule in Config.RULES
                for name in Config.ARCHITECTURES if name in models for track in tracks_for(name)]
        if Config.JOB_KEYS is not None:
            available = {job_key(*j) for j in jobs}
            unknown = set(Config.JOB_KEYS)-available
            if unknown:
                raise ValueError(f"本次可用队列中没有这些JOB_KEYS: {sorted(unknown)}")
            jobs = [j for j in jobs if job_key(*j) in Config.JOB_KEYS]
        atomic_json(out / "last_queue.json", [job_key(*j) for j in jobs])
        if not Config.SELF_TEST_ONLY:
            for index, (name, rule, track, rep) in enumerate(jobs):
                if should_stop(out):
                    break
                evaluator = SamplingEvaluator(name, models[name], data_by_rule[rule], track)
                try:
                    run_replica(out, evaluator, rule, rep, signature)
                except RuntimeError as exc:
                    # 初始化/恢复失败没有可用的新粒子状态，不能虚构一个K值。
                    key = job_key(name, rule, track, rep)
                    preflight_errors.append(dict(job=key, error=repr(exc), phase="initialization_or_restore"))
                    atomic_json(out / "preflight_errors.json", preflight_errors)
                    log(f"STARTUP ERROR {key}: {exc}；本次转到下一项")
                    torch.cuda.empty_cache()
                report = summarize(out, signature)
                log(f"PROGRESS 本次队列{index+1}/{len(jobs)} | 达标副本{report['completed_replicas']}/{report['planned_replicas']}")
        status = "self_test_only" if Config.SELF_TEST_ONLY else "paused" if should_stop(out) else "queue_finished"
    except KeyboardInterrupt:
        STOP, status = True, "paused"
    finally:
        try:
            error = sys.exc_info()[1]
            if error is not None:
                status = "error"
            report = summarize(out, signature)
            if status == "queue_finished" and report["completed_replicas"] == report["planned_replicas"]:
                status = "completed"
            atomic_json(out / "last_run_status.json", dict(status=status,
                        error=repr(error) if error else None, preflight_errors=preflight_errors,
                        completed_replicas=report["completed_replicas"], planned_replicas=report["planned_replicas"]))
            write_run_report(out, report, status)
            try:
                make_plots(out, report)
            except Exception as exc:
                atomic_json(out / "plot_error.json", dict(error=repr(exc)))
                log(f"绘图失败但数据完整保留：{exc}")
            log(f"FINAL {status} | 达标{report['completed_replicas']}/{report['planned_replicas']} | "
                "未完成项保留断点，下次继续")
            log(f"结果包：{package(out)}")
            log("续跑/续深请保留服务器上的整个结果目录；zip默认不含粒子检查点。")
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            lock.unlink(missing_ok=True)
            LOG_PATH = None
            RUN_STARTED = None


if __name__ == "__main__":
    main()
