"""小问题独立枚举和错误证明拒绝测试；仅使用CPU。"""

import itertools
import json
from pathlib import Path
import tempfile
import unittest
import uuid

from exact_circuits import CNF, atom_tables, apply_gate, check_assignment, decode_witness, encode, gate_clauses, check_lrat, replay_witness, solve


HERE = Path(__file__).resolve().parent


class ExactTests(unittest.TestCase):
    def setUp(self):
        self.folder = HERE/"test_artifacts"/str(uuid.uuid4())
        self.folder.mkdir(parents=True)

    def test_gate_encoding_all_values(self):
        for op in ("and","or","xor","not"):
            for a,b,o in itertools.product((False,True),repeat=3):
                cnf = CNF()
                selector,output = cnf.var(),cnf.var()
                gate_clauses(cnf,selector,output,op,a,b)
                expected = bool(apply_gate(op,int(a),int(b),1))
                values = [selector,output if o else -output]
                self.assertEqual(check_assignment(cnf,values),o == expected)

    def test_one_hot(self):
        cnf = CNF()
        variables = [cnf.var() for _ in range(4)]
        cnf.exactly_one(variables)
        for bits in itertools.product((False,True),repeat=4):
            possible = False
            for rest in itertools.product((False,True),repeat=cnf.variables-4):
                assignment = [i+1 if value else -(i+1) for i,value in enumerate(bits+rest)]
                possible |= check_assignment(cnf,assignment)
            self.assertEqual(possible,sum(bits)==1)

    def test_proof_and_corruption(self):
        cnf = CNF()
        x = cnf.var()
        cnf.add(x)
        cnf.add(-x)
        p,q = self.folder/"p.cnf",self.folder/"p.lrat"
        cnf.write(p)
        q.write_text("3 0 1 2 0\n",encoding="ascii")
        self.assertEqual(check_lrat(p,q)["additions"],1)
        q.write_text("3 0 1 0\n",encoding="ascii")
        with self.assertRaises(ValueError):
            check_lrat(p,q)
        q.write_text("3 0 1 -2 0\n",encoding="ascii")
        with self.assertRaises(ValueError):
            check_lrat(p,q)

    def test_real_solver_and_certificate(self):
        atoms,_ = atom_tables(3)
        a,b,c = atoms[2:]
        target = (a & b) | c
        cnf,meta = encode(3,[target],1,"aon_paid")
        no = solve(cnf,meta,self.folder/"one")
        self.assertEqual(no["status"],"unsat_verified")
        cnf,meta = encode(3,[target],2,"aon_paid")
        yes = solve(cnf,meta,self.folder/"two")
        self.assertEqual(yes["status"],"sat_verified")
        self.assertEqual(replay_witness(yes["witness"]),[target])

    def test_multioutput_share(self):
        atoms,_ = atom_tables(3)
        a,b,c = atoms[2:]
        for budget,expected in ((1,"unsat_verified"),(2,"sat_verified")):
            cnf,meta = encode(3,[a&b,(a&b)|c],budget,"aon_paid")
            result = solve(cnf,meta,self.folder/str(budget))
            self.assertEqual(result["status"],expected)

    def test_zero_gates_and_output_negation(self):
        atoms,_ = atom_tables(3)
        cnf,meta = encode(3,[atoms[1]^atoms[2],atoms[0]],0,"aon_paid")
        result = solve(cnf,meta,self.folder/"zero")
        self.assertEqual(result["status"],"sat_verified")


if __name__ == "__main__":
    unittest.main()
