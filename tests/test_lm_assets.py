"""Publisher identity, interrupted resume and no-Torch asset planning."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from tidegraph.lm.assets import download_file, verify_file
from tidegraph.lm.catalog import catalog, safe_relative


class Response(io.BytesIO):
    def __init__(self, data, status=200, **headers):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data)), **headers}


class Opener:
    def __init__(self, response):
        self.response, self.request = response, None

    def open(self, request, timeout):
        self.request = request
        return self.response


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/"weights.bin"
        self.data = b"independent asset payload"*100
        self.spec = dict(bytes=len(self.data),sha256=hashlib.sha256(self.data).hexdigest())

    def test_complete_and_resume(self):
        self.path.with_name("weights.bin.part").write_bytes(self.data[:91])
        opener = Opener(Response(self.data[91:],206,**{"Content-Range":f"bytes 91-{len(self.data)-1}/{len(self.data)}"}))
        download_file("https://example.test/asset",self.path,self.spec,lambda n:None,opener=opener)
        self.assertEqual(opener.request.get_header("Range"),"bytes=91-")
        self.assertTrue(verify_file(self.path,self.spec))
        self.assertFalse(self.path.with_name("weights.bin.part").exists())

    def test_server_ignores_resume(self):
        self.path.with_name("weights.bin.part").write_bytes(self.data[:91])
        with self.assertRaisesRegex(RuntimeError,"resume offset"):
            download_file("https://example.test/asset",self.path,self.spec,lambda n:None,opener=Opener(Response(self.data)))
        self.assertFalse(self.path.exists())
        self.assertEqual(self.path.with_name("weights.bin.part").stat().st_size,91)

    def test_hash_failure_never_publishes(self):
        bad = b"x"*len(self.data)
        with self.assertRaisesRegex(RuntimeError,"hash mismatch"):
            download_file("https://example.test/asset",self.path,self.spec,lambda n:None,opener=Opener(Response(bad)))
        self.assertFalse(self.path.exists())

    def test_git_blob_and_pinned_catalog(self):
        blob=hashlib.sha1(f"blob {len(self.data)}\0".encode()+self.data).hexdigest()
        self.path.write_bytes(self.data)
        self.assertTrue(verify_file(self.path,dict(bytes=len(self.data),git_blob=blob)))
        for value in catalog().values():
            self.assertRegex(value['revision'],r'^[0-9a-f]{40}$')
            for f in value['files']:
                safe_relative(f['path'])
                self.assertTrue(f.get('sha256') or f.get('git_blob'))
        for name in ['../escape','/absolute','a/../../escape']:
            with self.assertRaises(ValueError):safe_relative(name)


if __name__ == '__main__':
    unittest.main()
