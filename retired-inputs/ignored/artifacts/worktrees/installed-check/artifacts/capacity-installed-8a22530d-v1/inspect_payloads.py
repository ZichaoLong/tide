import collections,importlib.util,json,os,pathlib,sys,torch
job=pathlib.Path(__file__).parent
spec=importlib.util.spec_from_file_location("installed_harness",job/"check_installed_capacity.py")
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);h.torch=torch
out={}
for path in ("batched","placed"):
 for precision in ("fp32","bf16"):
  case=job/"results"/(path+"-"+precision)
  data=torch.load(case/"expected.pt",map_location="cpu",weights_only=False)
  record={}
  for label,value in (("parameters",data["updated"]["parameters"]),("gradients",data["gradients"]),("optimizer",data["updated"]["optimizer"]),("private_state",data["updated"]["state"])):
   tensors=collections.Counter();elements=collections.Counter();none=[]
   for name,item in h.leaves(value):
    if isinstance(item,torch.Tensor):tensors[str(item.dtype)]+=1;elements[str(item.dtype)]+=item.numel()
    elif item is None:none.append(name)
   record[label]=dict(tensor_counts=dict(tensors),elements=dict(elements),none_fields=none)
  record["nonfinite"]=h.finite(data);out[path+"-"+precision]=record
h.atomic(job/"payload-dtypes.json",out)
print(json.dumps(out))
