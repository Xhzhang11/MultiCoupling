# -*- coding: utf-8 -*-
import numpy as np
import torch
from scipy import interpolate
import sys
from matplotlib import pyplot as plt
class PlaceCells(object):
    def __init__(self, hp, us=None, is_cuda=False):

        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.hp = hp
        self.Np = hp['Np']
        self.sigma = hp['place_cell_rf']
        self.surround_scale = hp['surround_scale']
        self.box_width = hp['box_width']
        self.box_height = hp['box_height']
        self.is_periodic = hp['periodic']
        self.DoG = hp['DoG']
        self.softmax = torch.nn.Softmax(dim=-1)
        np.random.seed(0)# make the env maintain
        if hp['env'] == 'circle':
            self.c_recep_field = self.circle_env_even()

        elif hp['env'] == 'rectangle':
            self.c_recep_field = self.rectangle_env_even()



    def circle_env(self):
        np.random.seed(0)
        circle_center = (0.0, 0.0)  # Center of the circle (x_c, y_c)
        circle_radius = self.box_width/2  # Radius of the circle (r)
        # Generate random points within the circle
        angles = np.random.uniform(0, 2 * np.pi, self.Np)  # Random angles
        radii = np.sqrt(np.random.uniform(0, 1, self.Np)) * circle_radius  # Random radii
        usx = radii * np.cos(angles) + circle_center[0]  # x-coordinates
        usy = radii * np.sin(angles) + circle_center[1]  # y-coordinates
        # Combine into c_recep_field tensor
        c_recep_field = torch.tensor(np.vstack([usx, usy]).T).to(self.device)
        print('========== c_recep_field',c_recep_field.shape)

        # Rotation matrix
        rotation_angle = np.radians(self.hp['degree'])  # Rotation angle in radians (60 degrees)
        rotation_matrix = torch.tensor([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ]).to(self.device)

        print('===== rotation_matrix',rotation_matrix.shape)
        print('===== c_recep_field'  , c_recep_field.shape)

        # Apply rotation to all points
        if self.hp['rotate']:
            rotated_c_recep_field = c_recep_field @ rotation_matrix.T
        else:
            rotated_c_recep_field = c_recep_field

        # fig = plt.figure(figsize=(5, 5))
        # plt.plot(c_recep_field[:, 0], c_recep_field[:, 1], 'o', c='r', markersize=3,
        #          label='recep_field')
        #
        # plt.show()

        return rotated_c_recep_field

    def circle_env_even(self):
        circle_radius = self.box_width / 2

        # Step 1: Create a fine grid over the bounding square of the circle
        grid_spacing = self.box_width / np.ceil(np.sqrt(self.Np) * 1.1)
        x = np.arange(-circle_radius, circle_radius + grid_spacing, grid_spacing)
        y = np.arange(-circle_radius, circle_radius + grid_spacing, grid_spacing)
        xx, yy = np.meshgrid(x, y)
        xx = xx.flatten()
        yy = yy.flatten()

        # Step 2: Keep only points within the circle
        rr = np.sqrt(xx ** 2 + yy ** 2)
        mask = rr <= circle_radius
        xx = xx[mask]
        yy = yy[mask]

        # Step 3: Trim or pad to get exactly self.Np points
        total_points = len(xx)
        noise_scale = 0.02 # Small jitter to avoid perfect grid

        if total_points < self.Np:
            repeat = self.Np - total_points
            indices = np.random.choice(len(xx), repeat, replace=True)  # Randomly pick from valid points
            extra_xx = xx[indices] + np.random.normal(0, noise_scale, size=repeat)
            extra_yy = yy[indices] + np.random.normal(0, noise_scale, size=repeat)
            xx = np.concatenate([xx, extra_xx])
            yy = np.concatenate([yy, extra_yy])
        else:
            xx = xx[:self.Np]
            yy = yy[:self.Np]

        # Step 4: Add small noise to all points
        xx += np.random.normal(0, noise_scale, size=xx.shape)
        yy += np.random.normal(0, noise_scale, size=yy.shape)

        # Step 5: Convert to tensor
        c_recep_field_even = torch.tensor(np.vstack([xx, yy]).T, dtype=torch.float32).to(self.device)

        return c_recep_field_even


    def rectangle_env(self):
        # np.random.seed(0)
        usx = np.random.uniform(-self.box_width / 2, self.box_width / 2, (self.Np,))
        usy = np.random.uniform(-self.box_width / 2, self.box_width / 2, (self.Np,))
        # Combine into c_recep_field tensor
        c_recep_field = torch.tensor(np.vstack([usx, usy]).T).to(self.device)

        return c_recep_field

    def rectangle_env_even(self):
        grid_size = int(np.sqrt(self.Np))
        print('==grid_size',grid_size)
        print('self.box_width',self.box_width)
        x = np.linspace(-self.box_width / 2, self.box_width / 2, grid_size)
        y = np.linspace(-self.box_width / 2, self.box_width / 2, grid_size)
        xx, yy = np.meshgrid(x, y)
        xx = xx.flatten()
        yy = yy.flatten()

        # Add small noise to make it not perfectly regular
        noise_scale = 0.0
        xx += np.random.normal(0, noise_scale, size=xx.shape)
        yy += np.random.normal(0, noise_scale, size=yy.shape)

        # Trim to Np points in case grid_size^2 > Np
        xx = xx[:self.Np]
        yy = yy[:self.Np]

        # Create tensor
        c_recep_field_even = torch.tensor(np.vstack([xx, yy]).T).to(self.device)



        return c_recep_field_even


    def ramap_env(self):
        # # Generate random points within the box
        # usx = np.random.uniform(-self.box_width / 2, self.box_width / 2, (self.Np,))
        # usy = np.random.uniform(-self.box_width / 2, self.box_width / 2, (self.Np,))
        # c_recep_field = torch.tensor(np.vstack([usx, usy]).T).to(self.device)
        #
        # Create evenly spaced points within the box width
        grid_size = int(np.ceil(np.sqrt(self.Np)))  # Number of points per axis
        grid_x = np.linspace(-self.box_width / 2, self.box_width / 2, grid_size)
        grid_y = np.linspace(-self.box_width / 2, self.box_width / 2, grid_size)

        # Generate grid points
        xx, yy = np.meshgrid(grid_x, grid_y)
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T  # Flatten grid points

        # Select the first Np points (in turns)
        c_recep_field = torch.tensor(grid_points[:self.Np]).to(self.device)

        # Define a custom area: circle
        circle_center = (0.0, 0.0)  # Center of the circle (x_c, y_c)
        circle_radius = 0.9  # Radius of the circle (r)

        # Select points within the circular area
        circle_mask = ((c_recep_field[:, 0] - circle_center[0]) ** 2 +
                       (c_recep_field[:, 1] - circle_center[1]) ** 2) <= circle_radius ** 2
        c_recep_field = c_recep_field[circle_mask]  # Update to only include points inside the circle

        # Plot the updated points
        plt.scatter(c_recep_field[:, 0].cpu(), c_recep_field[:, 1].cpu(), c='red', s=20, label='Points in Circle')

        # Plot the circle boundary
        circle = plt.Circle(circle_center, circle_radius, color='blue', fill=False, linestyle='--', linewidth=1)
        plt.gca().add_artist(circle)

        # Configure plot
        plt.gca().set_aspect('equal')  # Ensure a circular appearance
        plt.xlim(-self.box_width / 2, self.box_width / 2)
        plt.ylim(-self.box_width / 2, self.box_width / 2)
        plt.title("Updated Points in Circular Area")
        plt.legend()
        plt.show()
        num_points_in_circle = c_recep_field.shape[0]
        return c_recep_field,num_points_in_circle



    def get_activation(self, x_pos):
        '''
        Get place cell activations for a given position.

        Args:
            x_pos: 2d position of shape [sequence_length, batch_size, 2].
            the number of all position is batch_size * sequence_length


        Returns:
            outputs: Place cell activations with shape [sequence_length, batch_size, Np].
        '''

        # print('x_pos',x_pos.shape,x_pos)

        # cue_position
        # sys.exit(0)


        # Flatten x_pos to 2D and calculate differences
        x_row = x_pos.shape[0]
        y_row = x_pos.shape[1]
        x_pos = x_pos.reshape(-1, 2)

        position = x_pos[:, None, :]
        c_recep_field = self.c_recep_field[None, :, :]

        differences = position - c_recep_field

        # print('====position',position.shape)
        # print('c_recep_field',c_recep_field.shape)
        # print('differences', differences.shape)


        # Reshape back to original structure if needed
        differences = differences.view(x_row, y_row, self.Np, 2)


        d = torch.abs(differences).float()
        norm2 = (d**2).sum(-1)


        # Normalize place cell outputs with prefactor alpha=1/2/np.pi/self.sigma**2,
        # or, simply normalize with softmax, which yields same normalization on
        # average and seems to speed up training.
        outputs = self.softmax(-norm2/(2*self.sigma**2))

        if self.DoG:
            # Again, normalize with prefactor
            # beta=1/2/np.pi/self.sigma**2/self.surround_scale, or use softmax.
            outputs -= self.softmax(-norm2/(2*self.surround_scale*self.sigma**2))
            # Shift and scale outputs so that they lie in [0,1].
            min_output,_ = outputs.min(-1,keepdims=True)
            outputs += torch.abs(min_output)
            outputs /= outputs.sum(-1, keepdims=True)


        return outputs

    def get_activation_test(self, x_pos):
        '''
        Get place cell activations for a given position.

        Args:
            x_pos: 2d position of shape [sequence_length, batch_size, 2].
            the number of all position is batch_size * sequence_length


        Returns:
            outputs: Place cell activations with shape [sequence_length, batch_size, Np].
        '''
        # Flatten x_pos to 2D and calculate differences
        x_row = x_pos.shape[0]
        y_row = x_pos.shape[1]
        x_pos = x_pos.reshape(-1, 2)
        differences = x_pos[:, None, :] - self.c_recep_field[None, :, :]
        # Reshape back to original structure if needed
        differences = differences.view(x_row, y_row, self.Np, 2)


        # print('======= x_pos',x_pos.shape,x_pos[:, :, None, :].shape)
        # print('self.c_recep_field',self.c_recep_field.shape,self.c_recep_field[None,None, ...].shape)
        # differences = x_pos[:, :, None, :] - self.c_recep_field[None,None, ...]

        d = torch.abs(differences).float()
        norm2 = (d**2).sum(-1)
        norm2_flat = norm2.reshape(-1)
        print('norm2_flat',norm2_flat.shape)




        # Normalize place cell outputs with prefactor alpha=1/2/np.pi/self.sigma**2,
        # or, simply normalize with softmax, which yields same normalization on
        # average and seems to speed up training.
        print('===norm2:', norm2.shape, norm2,2*self.sigma**2)
        outputs = self.softmax(-norm2/(2*self.sigma**2))
        outputs_flat = outputs.reshape(-1)
        print('outputs_flat','\n',outputs_flat)
        # plt.plot(norm2_flat, 'o-', c='r')
        # plt.plot(outputs_flat, 'o-', c='g')

        if self.DoG:
            # Again, normalize with prefactor
            # beta=1/2/np.pi/self.sigma**2/self.surround_scale, or use softmax.
            outputs -= self.softmax(-norm2/(2*self.surround_scale*self.sigma**2))

            # Shift and scale outputs so that they lie in [0,1].
            min_output,_ = outputs.min(-1,keepdims=True)
            outputs += torch.abs(min_output)
            outputs /= outputs.sum(-1, keepdims=True)


        outputs_flat_1 = outputs.reshape(-1)
        print('@@@ outputs','\n',outputs_flat_1)

        # plt.plot(outputs_flat_1, '*-', c='b')
        # plt.show()




        return outputs

    def get_nearest_cell_pos(self, activation):
        '''
        Decode position using centers of k maximally active place cells.
        
        Args: 
            activation: Place cell activations of shape [batch_size, sequence_length, Np].
            k: Number of maximally active place cells with which to decode position.

        Returns:
            pred_pos: Predicted 2d position with shape [batch_size, sequence_length, 2].
        '''



        k = self.hp['k']
        value_max_k, idxs = torch.topk(activation, k=k)
        # print('** idxs',idxs[:,:,0].shape)

        #print('** value_max_k',value_max_k.shape,'\n',value_max_k)
        pred_pos = self.c_recep_field[idxs].mean(-2)

        us_x = self.c_recep_field[:,0]


        cell_idx = idxs[:,:,0]#torch.Size([10, 512])
        # print('cell_idx',cell_idx[:,0])

        return pred_pos

    def get_nearest_cell_pos_find_sequence(self, activation):
        '''
        Decode position using centers of k maximally active place cells.

        Args:
            activation: Place cell activations of shape [batch_size, sequence_length, Np].
            k: Number of maximally active place cells with which to decode position.

        Returns:
            pred_pos: Predicted 2d position with shape [batch_size, sequence_length, 2].
        '''

        # print('activation',activation.shape,activation)

        k = self.hp['k']
        value_max_k, idxs = torch.topk(1*activation, k=k)
        print('** idxs', idxs[:, :, 0].shape)

        # print('** value_max_k',value_max_k.shape,'\n',value_max_k)
        pred_pos = self.c_recep_field[idxs].mean(-2)

        us_x = self.c_recep_field[:, 0]

        cell_idx = idxs[:, :, 0]  # torch.Size([10, 512])
        # print('cell_idx', cell_idx[:, 0])

        return pred_pos,cell_idx

    def grid_pc(self, pc_outputs, res=32):
        ''' Interpolate place cell outputs onto a grid'''
        coordsx = np.linspace(-self.box_width/2, self.box_width/2, res)
        coordsy = np.linspace(-self.box_height/2, self.box_height/2, res)
        grid_x, grid_y = np.meshgrid(coordsx, coordsy)
        grid = np.stack([grid_x.ravel(), grid_y.ravel()]).T

        # Convert to numpy
        pc_outputs = pc_outputs.reshape(-1, self.Np)
        print('****pc_outputs',pc_outputs.shape)
        print()
        
        T = pc_outputs.shape[0] #T vs transpose? What is T? (dim's?)
        print("T",T)
        pc = np.zeros([T, res, res])
        for i in range(len(pc_outputs)):
            # print('self.c_recep_field.shape',self.c_recep_field.shape)
            # print('pc_outputs[i].shape',pc_outputs[i].shape)
            gridval = interpolate.griddata(self.c_recep_field, pc_outputs[i], grid,method='linear', fill_value=np.nan, rescale=False)
            pc[i] = gridval.reshape([res, res])
        
        return pc

    def compute_covariance(self, res=30):
        '''Compute spatial covariance matrix of place cell outputs'''
        pos = np.array(np.meshgrid(np.linspace(-self.box_width/2, self.box_width/2, res),
                         np.linspace(-self.box_height/2, self.box_height/2, res))).T


        pos = torch.tensor(pos, device=self.device)

        #Maybe specify dimensions here again?
        #pc_outputs = self.get_activation(pos).reshape(-1,self.Np).cpu()
        #pc_outputs = self.get_activation(pos).reshape(-1,self.Np).cuda()

        pc_outputs = self.get_activation(pos).reshape(-1,self.Np)
        self.pc_outputs_for_plot = pc_outputs

        C = np.matmul(pc_outputs,pc_outputs.T)  #pc_outputs@pc_outputs.T

        Csquare = C.reshape(res,res,res,res)

        Cmean = np.zeros([res,res])
        for i in range(res):
            for j in range(res):
                Cmean += np.roll(np.roll(Csquare[i,j], -i, axis=0), -j, axis=1)
                
        Cmean = np.roll(np.roll(Cmean, res//2, axis=0), res//2, axis=1)


        '''
        fig = plt.figure()
        for i in range(9):
            plt.plot(pc_outputs[i,:])
        fig = plt.figure()
        plt.imshow(self.pc_outputs_for_plot, cmap='jet', interpolation='gaussian')
        
        fig = plt.figure()
        plt.imshow(C, cmap='jet', interpolation='gaussian')
        
        fig = plt.figure()
        plt.imshow(Cmean, cmap='jet', interpolation='gaussian')
        '''
        return Cmean

if __name__ == '__main__':
    from defaults import get_default_hp
    from matplotlib import pyplot as plt

    #load parames
    hp = get_default_hp()
    hp['Np']=512
    hp['new_env']=False
    hp['env'] = 'circle'#rectangle,circle
    hp['rotate']=True
    res=3
    place_cells = PlaceCells(hp)







    #============================= plot ==============================
    fig, ax = plt.subplots(figsize=(5, 5))
    # position
    plt.plot(place_cells.c_recep_field[:,0],place_cells.c_recep_field[:,1],'o',c='grey',markersize=3,label='recep_field')
    # plt.plot(place_cells.c_recep_field[0,0],place_cells.c_recep_field[0,1],'o',c='g',markersize=8,label='recep_field')
    # plt.plot(place_cells.c_recep_field[1,0],place_cells.c_recep_field[1,1],'o',c='b',markersize=8,label='recep_field')
    # Big circle boundary (environment)
    outer_circle = plt.Circle((0, 0), hp['box_width'] / 2, fill=False, linestyle='--', color='gray',
                              label='Env boundary')
    ax.add_patch(outer_circle)

    # Small circle with radius 0.1

    select_point_x = place_cells.c_recep_field[200,0]
    select_point_y = place_cells.c_recep_field[200,1]
    small_circle = plt.Circle((select_point_x, select_point_y), 0.13, fill=False, color='blue', linewidth=2, label='Radius=0.1')
    plt.plot(select_point_x,select_point_y,'o',c='r',markersize=3,label='recep_field')


    ax.add_patch(small_circle)
    # Configure plot
    plt.gca().set_aspect('equal')  # Ensure a circular appearance
    plt.xlim(-1.3 - 0.1, 1.3 + 0.1)
    plt.ylim(-1.3 - 0.1, 1.3 + 0.1)
    plt.title("Points Directly Generated in Circular Area")
    #plt.legend()
    plt.show()












