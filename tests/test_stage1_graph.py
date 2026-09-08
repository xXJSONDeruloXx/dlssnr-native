import ctypes as c
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'native'


class Stage1GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler=os.environ.get('CC','cc')
        if not shutil.which(compiler):
            raise unittest.SkipTest('C compiler unavailable')
        cls.temp=tempfile.TemporaryDirectory()
        cls.library=Path(cls.temp.name)/'libstage1-graph-test.so'
        subprocess.run([
            compiler,'-O2','-std=c11','-Wall','-Wextra','-Werror','-fPIC','-shared',
            '-I'+str(NATIVE),str(NATIVE/'stage1_graph.c'),'-o',str(cls.library)
        ],check=True)
        cls.lib=c.CDLL(str(cls.library))
        cls.lib.nr_stage1_required_tensor_count.restype=c.c_uint32
        cls.lib.nr_stage1_required_tensor.argtypes=[
            c.c_uint32,c.c_char_p,c.c_size_t,c.POINTER(c.c_int)]
        cls.lib.nr_stage1_required_tensor.restype=c.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def tensor(self,index):
        name=c.create_string_buffer(64);role=c.c_int()
        self.assertEqual(self.lib.nr_stage1_required_tensor(
            index,name,len(name),c.byref(role)),0)
        return name.value.decode(),role.value

    def test_inventory_is_unique_and_complete(self):
        count=self.lib.nr_stage1_required_tensor_count()
        self.assertEqual(count,152)
        entries=[self.tensor(i) for i in range(count)]
        names=[name for name,_ in entries]
        self.assertEqual(len(set(names)),152)
        self.assertEqual(names[0],'block0.layer0.layer')
        self.assertIn('block30.layer4.layer',names)
        self.assertIn('block38.layer4.layer',names)
        self.assertEqual(names[-1],'block70.layer0.layer')

    def test_expected_stage_boundaries(self):
        names=[self.tensor(i)[0] for i in range(152)]
        for block in range(1,23):
            self.assertIn(f'block{block}.layer0.layer',names)
        for block in range(23,31):
            for layer in range(4):
                self.assertIn(f'block{block}.layer{layer}.layer',names)
        for block in range(31,39):
            for layer in range(5):
                self.assertIn(f'block{block}.layer{layer}.layer',names)
        for block in range(40,48):
            for layer in range(4):
                self.assertIn(f'block{block}.layer{layer}.layer',names)


if __name__=='__main__':unittest.main()
