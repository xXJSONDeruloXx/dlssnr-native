import ctypes as c
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'native'


class NativeModelPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which(os.environ.get('CC','cc')):
            raise unittest.SkipTest('C compiler unavailable')
        cls.temp=tempfile.TemporaryDirectory()
        cls.library=Path(cls.temp.name)/'libmodel-package-test.so'
        subprocess.run([
            os.environ.get('CC','cc'),'-O2','-std=c11','-Wall','-Wextra','-Werror','-fPIC','-shared',
            '-I'+str(NATIVE),str(NATIVE/'model_package.c'),str(NATIVE/'sha256.c'),'-o',str(cls.library)
        ],check=True)
        cls.lib=c.CDLL(str(cls.library))
        cls.lib.nr_model_package_open.argtypes=[c.c_char_p,c.POINTER(c.c_void_p)]
        cls.lib.nr_model_package_open.restype=c.c_int
        cls.lib.nr_model_package_close.argtypes=[c.c_void_p]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def open_result(self,data):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.dlssnr'
            path.write_bytes(data)
            handle=c.c_void_p()
            result=self.lib.nr_model_package_open(os.fsencode(path),c.byref(handle))
            if handle.value:self.lib.nr_model_package_close(handle)
            return result

    def test_truncated_package_rejected(self):
        self.assertNotEqual(self.open_result(b'DLSSNRM1'),0)

    def test_wrong_profile_rejected(self):
        header=struct.pack('<8sII32s32sQ',b'DLSSNRM1',1,153,bytes(32),bytes(32),128)
        self.assertNotEqual(self.open_result(header+bytes(40)),0)

    @unittest.skipUnless(os.environ.get('DLSSNR_TEST_PACKAGE'),'optional private package fixture')
    def test_private_package_opens(self):
        handle=c.c_void_p()
        result=self.lib.nr_model_package_open(
            os.fsencode(Path(os.environ['DLSSNR_TEST_PACKAGE']).resolve()),c.byref(handle))
        self.assertEqual(result,0)
        self.assertTrue(handle.value)
        self.lib.nr_model_package_close(handle)


if __name__=='__main__':unittest.main()
