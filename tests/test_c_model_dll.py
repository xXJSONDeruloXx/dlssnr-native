import ctypes as c
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'native'


class NativeDllModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler=os.environ.get('CC','cc')
        if not shutil.which(compiler):
            raise unittest.SkipTest('C compiler unavailable')
        cls.temp=tempfile.TemporaryDirectory()
        cls.library=Path(cls.temp.name)/'libdll-model-test.so'
        subprocess.run([
            compiler,'-O2','-std=c11','-Wall','-Wextra','-Werror','-fPIC','-shared',
            '-I'+str(NATIVE),str(NATIVE/'model_dll.c'),str(NATIVE/'sha256.c'),
            '-o',str(cls.library)
        ],check=True)
        cls.lib=c.CDLL(str(cls.library))
        cls.lib.nr_model_dll_open.argtypes=[c.c_char_p,c.POINTER(c.c_void_p)]
        cls.lib.nr_model_dll_open.restype=c.c_int
        cls.lib.nr_model_dll_close.argtypes=[c.c_void_p]
        cls.lib.nr_model_dll_tensor_count.argtypes=[c.c_void_p]
        cls.lib.nr_model_dll_tensor_count.restype=c.c_uint32
        cls.lib.nr_model_dll_weights_sha256.argtypes=[c.c_void_p]
        cls.lib.nr_model_dll_weights_sha256.restype=c.c_char_p

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_unrecognized_binary_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.dll'
            path.write_bytes(b'MZ'+bytes(256))
            handle=c.c_void_p()
            self.assertNotEqual(self.lib.nr_model_dll_open(os.fsencode(path),c.byref(handle)),0)
            self.assertFalse(handle.value)

    @unittest.skipUnless(os.environ.get('DLSSNR_PORTABLE_TEST_MODEL'),'optional portable-profile DLL')
    def test_private_dll_opens_directly(self):
        handle=c.c_void_p()
        path=Path(os.environ['DLSSNR_PORTABLE_TEST_MODEL']).resolve()
        result=self.lib.nr_model_dll_open(os.fsencode(path),c.byref(handle))
        self.assertEqual(result,0)
        try:
            self.assertEqual(self.lib.nr_model_dll_tensor_count(handle),153)
            self.assertEqual(
                self.lib.nr_model_dll_weights_sha256(handle).decode(),
                'c7f86ab233356fe73d7f66b749559c85838cdce3149b228480154801dff8bc60')
        finally:
            self.lib.nr_model_dll_close(handle)


if __name__=='__main__':unittest.main()
