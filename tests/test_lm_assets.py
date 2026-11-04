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
        self.assertEqual(opener.request.get_header("Range"),f"bytes=91-{len(self.data)-1}")
        self.assertTrue(verify_file(self.path,self.spec))
        self.assertFalse(self.path.with_name("weights.bin.part").exists())

    def test_server_ignores_resume(self):
        self.path.with_name("weights.bin.part").write_bytes(self.data[:91])
        with self.assertRaisesRegex(RuntimeError,"requested Range"):
            download_file("https://example.test/asset",self.path,self.spec,lambda n:None,opener=Opener(Response(self.data)))
        self.assertFalse(self.path.exists())
        self.assertEqual(self.path.with_name("weights.bin.part").stat().st_size,91)

    def test_segmented_read_resumes_after_early_eof(self):
        requests=[]; events=[]; data=self.data
        class Ranges:
            def open(self,request,timeout):
                start,end=map(int,request.get_header('Range')[6:].split('-'))
                requests.append((start,end))
                body=data[start:end+1]
                if len(requests)==2:body=body[:17]
                return Response(body,206,**{'Content-Range':f'bytes {start}-{end}/{len(data)}',
                                            'Content-Length':str(end-start+1)})
        download_file('https://example.test/file',self.path,self.spec,lambda n:None,
                      opener=Ranges(),chunk_bytes=128,retries=1,event=events.append,sleep=lambda _:None)
        self.assertTrue(verify_file(self.path,self.spec))
        self.assertEqual(requests[:3],[(0,127),(128,255),(145,272)])
        self.assertEqual(events[0]['offset'],145)

    def test_wrong_range_total_is_not_retried(self):
        response=Response(self.data[:128],206,**{'Content-Range':f'bytes 0-127/{len(self.data)+1}'})
        with self.assertRaisesRegex(RuntimeError,'requested Range'):
            download_file('https://example.test/file',self.path,self.spec,lambda n:None,
                          opener=Opener(response),chunk_bytes=128)
        self.assertFalse(self.path.exists())

    def test_retries_exhaust_and_keep_prefix(self):
        class Empty:
            def open(inner,request,timeout):
                return Response(b'',206,**{'Content-Range':f'bytes 0-127/{len(self.data)}','Content-Length':'128'})
        events=[]
        with self.assertRaisesRegex(RuntimeError,'budget exhausted'):
            download_file('https://example.test/file',self.path,self.spec,lambda n:None,
                          opener=Empty(),chunk_bytes=128,retries=1,event=events.append,sleep=lambda _:None)
        self.assertEqual(len(events),2)
        self.assertFalse(self.path.exists())
        self.assertEqual(self.path.with_name("weights.bin.part").stat().st_size,0)

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


def test_fetch_retains_previous_failure_in_new_attempt_directory(tmp_path, monkeypatch):
    from tidegraph.lm import assets
    from tidegraph.experiment.records import write_json, read_json, fingerprint
    destination=tmp_path/'model';destination.mkdir()
    previous=dict(state='failed',error='early EOF',files={})
    write_json(destination/'asset-state.json',previous)
    spec=dict(id='test/model',revision='a'*40,files=[dict(path='weights.bin',bytes=3,sha256=hashlib.sha256(b'abc').hexdigest())])
    monkeypatch.setattr(assets,'model',lambda _:spec)
    monkeypatch.setattr(assets,'local_path',lambda *_:destination)
    def download(url,path,*args,**kwargs):
        path.write_bytes(b'abc');return 'downloaded_verified'
    monkeypatch.setattr(assets,'download_file',download)
    result=assets.fetch('test',tmp_path,tmp_path)
    assert result['state']=='complete'
    assert read_json(destination/'attempts'/(fingerprint(previous)+'.json'))==previous


def test_connection_timeout_is_explicit_and_error_does_not_leak_urls(tmp_path):
    import urllib.error
    import pytest
    from tidegraph.lm.asset_transport import network_error
    class Failing:
        def open(self,request,timeout):
            assert timeout==120
            raise urllib.error.URLError(OSError(101,'private signed URL'))
    events=[]
    with pytest.raises(RuntimeError,match='retry budget exhausted'):
        download_file('https://example.test/private',tmp_path/'file',dict(bytes=2,sha256='unused'),
                      lambda _:None,opener=Failing(),retries=0,timeout_seconds=120,event=events.append)
    assert events[0]['reason']=='URLError/OSError/errno=101'
    assert 'private' not in network_error(urllib.error.URLError('private credentials'))
    assert 'private' not in repr(events)
