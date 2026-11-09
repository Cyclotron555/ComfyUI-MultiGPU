import os,sys,unittest,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--comfy-root',type=Path,required=True);args=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(args.comfy_root.resolve()))
import comfy.options
comfy.options.enable_args_parsing();sys.argv=[sys.argv[0],'--cpu','--offline']
import torch
from comfy_extras.nodes_wan import WanVaceToVideo
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wan_vace_reference import prepare_conditioning
class Tests(unittest.TestCase):
 def equivalent(self,ref):
  a=torch.full((1,16,2,2,2),.125);b=a+.2;r=torch.full((1,16,1,2,2),.75)
  pos=[[torch.zeros(1,2,4),{'marker':'original'}]];neg=[[torch.ones(1,2,4),{}]]
  queue=iter([r,a,b] if ref else [a,b])
  class VAE:
   def encode(self,pixels):return next(queue).clone()
  native=WanVaceToVideo.execute(pos,neg,VAE(),16,16,5,1,1.,control_video=torch.zeros(5,16,16,3),reference_image=torch.zeros(1,16,16,3) if ref else None)
  actual=prepare_conditioning(pos,neg,a,b,frames=5,width=16,height=16,reference=r if ref else None)
  self.assertEqual(actual[3],native[3]);self.assertTrue(torch.equal(actual[2]['samples'],native[2]['samples']))
  for i in [0,1]:
   for key in ['vace_frames','vace_mask']:
    self.assertTrue(torch.equal(actual[i][0][1][key][0],native[i][0][1][key][0]),key)
   self.assertEqual(actual[i][0][1]['vace_strength'],native[i][0][1]['vace_strength'])
  self.assertNotIn('vace_frames',pos[0][1])
 def test_without_reference(self):self.equivalent(False)
 def test_with_reference(self):self.equivalent(True)
 def test_native_non_ui_geometry_and_strength_are_preserved(self):
  a=torch.zeros(1,16,1,2,3);pos=[[torch.zeros(1,2,4),{}]]
  queue=iter([a,a])
  class VAE:
   def encode(self,pixels):return next(queue).clone()
  native=WanVaceToVideo.execute(pos,pos,VAE(),24,16,3,1,3.,control_video=torch.zeros(3,16,24,3))
  actual=prepare_conditioning(pos,pos,a,a,frames=3,width=24,height=16,strength=3.)
  self.assertTrue(torch.equal(actual[0][0][1]['vace_mask'][0],native[0][0][1]['vace_mask'][0]))
  self.assertEqual(actual[0][0][1]['vace_strength'],[3.])
  self.assertTrue(torch.equal(actual[2]['samples'],native[2]['samples']))
 def test_invalid_shapes_and_nan_rejected(self):
  a=torch.zeros(1,16,2,2,2)
  for x in [a[:,:,:1],torch.full_like(a,float('nan'))]:
   with self.assertRaises(ValueError):prepare_conditioning([],[],x,a,frames=5,width=16,height=16)
 def test_unsupported_geometry_and_strength_rejected(self):
  a=torch.zeros(1,16,2,2,2)
  for kwargs in [dict(frames=5,width=15,height=16),dict(frames=5,width=16,height=16,strength=float('nan'))]:
   with self.assertRaises(ValueError):prepare_conditioning([],[],a,a,**kwargs)
unittest.main(argv=['test'],verbosity=2)
