import torch
from scene import Scene
import torch.optim as optim
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.calculate_error_utils import cal_campose_error
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams,iComMaParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.icomma_helper import load_LoFTR, get_pose_estimation_input, get_pose_estimation_input_2
from utils.image_utils import to8b
import cv2
import imageio
import os
import ast
from scene.cameras import Camera_Pose
from utils.loss_utils import loss_loftr,loss_mse

from PIL import Image
from typing import NamedTuple
import numpy as np
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat, rotmat2qvec
from utils.icomma_helper import combine_3dgs_rotation_translation, trans_t_xyz, rot_phi, rot_psi, rot_theta



def camera_pose_estimation_2(gaussians:GaussianModel, background:torch.tensor, pipeline:PipelineParams, icommaparams:iComMaParams, icomma_info, output_path, LoFTR_model):
    # start pose 
    start_pose_w2c=icomma_info.start_pose_w2c.cuda()
    
    # query_image for comparing 
    query_image = icomma_info.query_image.cuda()

    # initialize camera pose object
    camera_pose = Camera_Pose(start_pose_w2c,FoVx=icomma_info.FoVx,FoVy=icomma_info.FoVy,
                            image_width=icomma_info.image_width,image_height=icomma_info.image_height)
    camera_pose.cuda()

    print(f" Initial_camera_pose: {camera_pose.pose_w2c}")

    # store gif elements
    imgs=[]
    
    matching_flag= not icommaparams.deprecate_matching

    # start optimizing
    optimizer = optim.Adam(camera_pose.parameters(),lr = icommaparams.camera_pose_lr)
    iter = icommaparams.pose_estimation_iter
    num_iter_matching = 0
    for k in range(iter):

        rendering = render(camera_pose,gaussians, pipeline, background,compute_grad_cov2d = icommaparams.compute_grad_cov2d)["render"]

        if matching_flag:
            #print("Matching flag is ON")
            loss_matching = loss_loftr(query_image,rendering,LoFTR_model,icommaparams.confidence_threshold_LoFTR,icommaparams.min_matching_points)
            loss_comparing = loss_mse(rendering,query_image)
            
            if loss_matching is None:
                loss = loss_comparing
            else:  
                loss = icommaparams.lambda_LoFTR *loss_matching + (1-icommaparams.lambda_LoFTR)*loss_comparing
                if loss_matching<0.001:
                    matching_flag=False
                    
            num_iter_matching += 1
        else:
            #print("Matching flag is OFF")
            loss_comparing = loss_mse(rendering,query_image)
            loss = loss_comparing
            
            new_lrate = icommaparams.camera_pose_lr * (0.6 ** ((k - num_iter_matching + 1) / 50))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
        
        # output intermediate results
        if (k + 1) % 50 == 0 or k == 0:
            #print(f"Matching flag is {matching_flag}")
            print('Step: ', k)
            if matching_flag and loss_matching is not None:
                print('Matching Loss: ', loss_matching.item())
            print('Comparing Loss: ', loss_comparing.item())
            print('Loss: ', loss.item())
               
            # output images
            if icommaparams.OVERLAY is True:
                with torch.no_grad():
                    rgb = rendering.clone().permute(1, 2, 0).cpu().detach().numpy()
                    rgb8 = to8b(rgb)
                    ref = to8b(query_image.permute(1, 2, 0).cpu().detach().numpy())
                    filename = os.path.join(output_path, str(file_id)+'- '+str(k)+'.png')
                    dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
                    imageio.imwrite(filename, dst)
                    imgs.append(dst)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        camera_pose(start_pose_w2c)
        

    # output gif
    print(f"Final camera_pose: {camera_pose.pose_w2c}")
    # if icommaparams.OVERLAY is True:
    #     imageio.mimwrite(os.path.join(output_path, str(file_id)+'video.gif'), imgs, fps=4)

    return camera_pose.pose_w2c

class iComMa_input_info(NamedTuple):
    start_pose_w2c:torch.tensor
    gt_pose_c2w:np.array
    query_image:torch.tensor
    FoVx:float
    FoVy:float
    image_width:int
    image_height:int

class poseEstimation:
  
#   img_path='/home/siva_2204/image_data_bk/test_image/00003.png',
#   delta="[30,10,10,0.1,0.1,0.1]",

    def __init__(self, fx=580.58204039920929, fy=580.58204039920929, height=718, width=1078, iteration=30000, quiet=False,              
                model_path='/home/siva_2204/simulation_data2/output', output_path='output_icomma' ):

        self.output_path = output_path
        self.width = width
        self.height = height
        
        # Set up command line argument parser
        parser = ArgumentParser(description="Camera pose estimation parameters")

        #Create args with standalone defaults (NEW)
        args = Namespace( )

        model = ModelParams(parser, sentinel=True)
        model.extract(args)
        self.pipeline = PipelineParams(parser).extract(args)
        self.icommaparams = iComMaParams(parser).extract(args)        

        #print(f"args are: {args}")
    
        # Initialize system state (RNG)
        safe_state(quiet)

        makedirs(self.output_path, exist_ok=True)
        
        # load LoFTR_model
        self.LoFTR_model=load_LoFTR(self.icommaparams.LoFTR_ckpt_path,self.icommaparams.LoFTR_temp_bug_fix)
        -1.689510,-2.985026,0.016072
        # load gaussians
        dataset = model.extract(args)

        self.gaussians = GaussianModel(dataset.sh_degree)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # get camera info from Scene
        # Reused 3DGS code to obtain camera information. 
        # You can customize the iComMa_input_info in practical applications.
        Scene( model_path,self.gaussians,load_iteration=iteration,shuffle=False)
    

        #------------construct icomma_info object----------------
        self.FovY = focal2fov(fy, height)
        self.FovX = focal2fov(fx, width)  


    def find_pose(self, img_path='/home/siva_2204/image_data_bk/test_image/00003.png', qvec =[1,2,3,4], tvec =[3,4,4], delta="[30,10,10,0.1,0.1,0.1]"):
        
        global flag

        delta = ast.literal_eval(delta)
        print(f"delta_1: {delta}")
        R = np.transpose(qvec2rotmat(qvec))
        T = np.array(tvec)
        
        gt_pose_c2w=combine_3dgs_rotation_translation(R,T)
         
        start_pose_c2w = gt_pose_c2w # this is the estimated pose for previous image .
        #start_pose_c2w =  trans_t_xyz(delta[3],delta[4],delta[5]) @ rot_phi(delta[0]/180.*np.pi) @ rot_theta(delta[1]/180.*np.pi) @ rot_psi(delta[2]/180.*np.pi)  @ gt_pose_c2w
        #print(f"start_pose_c2w:{start_pose_c2w}")

        PIL_image = Image.open(img_path)
        image = torch.from_numpy(np.array(PIL_image)) / 255.0
        if len(image.shape) == 3:
            image = image.permute(2, 0, 1)
        else:
            image = image.unsqueeze(dim=-1).permute(2, 0, 1) 

        start_pose_w2c =None
        if flag ==1:
            print("in Flag condition: first time")  
            start_pose_w2c = torch.from_numpy(np.linalg.inv(start_pose_c2w)).float()
            #flag =0
        else:
            start_pose_w2c = torch.from_numpy(start_pose_c2w).float()

        print(f"start pose C2W is :{start_pose_c2w}")        
        
        icomma_info = iComMa_input_info(gt_pose_c2w=None,
            start_pose_w2c= start_pose_w2c,   #torch.from_numpy(np.linalg.inv(start_pose_c2w)).float(),
            query_image= image,
            FoVx=self.FovX,
            FoVy=self.FovY,
            image_width=self.width,
            image_height=self.height)

        #icomma_info=get_pose_estimation_input_3(image, FovX, FovY, ast.literal_eval(args.delta))
        # print("-------------icomma_info object--------------------------")
        # print(f"start_pose W2C: {icomma_info.start_pose_w2c}")
        # print(f"gt_pose_c2w: {icomma_info.gt_pose_c2w}")
        # print(f"FoVx: {icomma_info.FoVx}")
        # print(f"FoVy: {icomma_info.FoVy}")
        # print(f"image_width: {icomma_info.image_width}")
        # print(f"image_height:{icomma_info.image_height}")

        # print("-------------icomma_info object--------------------------")

        result = camera_pose_estimation_2(self.gaussians,self.background,self.pipeline,self.icommaparams, icomma_info, self.output_path, self.LoFTR_model)
        return result
    
    def run (self, image_folder_path = '/home/siva_2204/gazebo_images_data/sampled_test_path_images'):
        
        global file_id        

        image_files = sorted(
            [f for f in os.listdir(image_folder_path) if f.endswith('.jpg')],
            key=lambda x: int(x.split('.')[0]) ) # Sort by the number in the filename

        Q =[-0.46, 0.00, 0.89, 0.00]
        T = [1.91, 0, 3]
        for image in image_files:
            file_id =image
            image_path = os.path.join(image_folder_path, image)
            print(f"estimating pose for {image_path}")
            result = self.find_pose(img_path=image_path, qvec =Q, tvec =T)
            result = result.detach().cpu().numpy()
            print(f"1.type of result = {type(result)}")
           
            Q = rotmat2qvec(result[:3, :3])
            T =result[:3, 3]
            T = T.reshape(1, 3)



if __name__ == "__main__":
    flag =1
    file_id =None
    obj = poseEstimation(fx=580.58204039920929, fy=580.58204039920929, height=718, width=1078, iteration=30000, quiet=False,              
                model_path='/home/siva_2204/simulation_data2/output', output_path='output_icomma')
    obj.run(image_folder_path = '/home/siva_2204/gazebo_images_data/sampled_test_path_images')
   
