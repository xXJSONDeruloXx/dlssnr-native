import ctypes as c
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'native'


class NativeModelDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler=os.environ.get('CC','cc')
        if not shutil.which(compiler):
            raise unittest.SkipTest('C compiler unavailable')
        cls.temp=tempfile.TemporaryDirectory()
        cls.library=Path(cls.temp.name)/'libdecode-test.so'
        subprocess.run([
            compiler,'-O2','-std=c11','-Wall','-Wextra','-Werror','-fPIC','-shared',
            '-I'+str(NATIVE),str(NATIVE/'model_decode.c'),'-lm','-o',str(cls.library)
        ],check=True)
        cls.lib=c.CDLL(str(cls.library))
        cls.lib.nr_e4m3fn_decode.argtypes=[c.c_uint8]
        cls.lib.nr_e4m3fn_decode.restype=c.c_float
        cls.lib.nr_fp16_decode.argtypes=[c.c_uint16]
        cls.lib.nr_fp16_decode.restype=c.c_float
        cls.lib.nr_tin_channel_permutation.argtypes=[c.c_uint32]
        cls.lib.nr_tin_channel_permutation.restype=c.c_uint32
        cls.lib.nr_unpack_matrix_fp8.argtypes=[
            c.c_void_p,c.c_size_t,c.c_uint32,c.c_uint32,c.c_int,c.POINTER(c.c_float),c.c_size_t]
        cls.lib.nr_unpack_matrix_fp8.restype=c.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_numeric_formats(self):
        self.assertEqual(self.lib.nr_e4m3fn_decode(0x38),1.0)
        self.assertEqual(self.lib.nr_e4m3fn_decode(0x40),2.0)
        self.assertEqual(self.lib.nr_e4m3fn_decode(0xb8),-1.0)
        self.assertEqual(self.lib.nr_e4m3fn_decode(0x7e),448.0)
        self.assertTrue(math.isnan(self.lib.nr_e4m3fn_decode(0x7f)))
        self.assertEqual(self.lib.nr_fp16_decode(0x3c00),1.0)
        self.assertEqual(self.lib.nr_fp16_decode(0xc000),-2.0)
        self.assertEqual(self.lib.nr_fp16_decode(0x0001),2.0**-24)
        self.assertTrue(math.isinf(self.lib.nr_fp16_decode(0x7c00)))

    def test_tin_permutation_is_bijective(self):
        for channels in (32,64,128,256,512,1024):
            values=[self.lib.nr_tin_channel_permutation(i) for i in range(channels)]
            self.assertEqual(sorted(values),list(range(channels)))

    def test_fp8_matrix_layout_known_positions(self):
        rows,columns=16,32
        raw=(c.c_uint8*(rows*columns))()
        def index(row,column):
            nb,rr=divmod(row,16);d4,d2=divmod(rr,8)
            kb,cc=divmod(column,32);d5,rest=divmod(cc,8);d3,d6=divmod(rest,2)
            return kb*(rows//16)*512+nb*512+d2*64+d3*16+d4*8+d5*2+d6
        points=[(0,0),(1,1),(7,31),(8,0),(15,31)]
        for row,column in points:raw[index(row,column)]=0x38
        output=(c.c_float*(rows*columns))()
        self.assertEqual(self.lib.nr_unpack_matrix_fp8(
            raw,len(raw),rows,columns,0,output,len(output)),0)
        for row in range(rows):
            for column in range(columns):
                expected=1.0 if (row,column) in points else 0.0
                self.assertEqual(output[row*columns+column],expected)


if __name__=='__main__':unittest.main()
