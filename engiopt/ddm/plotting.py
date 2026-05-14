"""
    Plotting functions for the diffusion model.

"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import scienceplots
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.style.use('science')

def show_tensor_image(image, cpw_mode=False, s=0.5, ax=None):

    # Take first image of batch
    d = 3

    if len(image.shape) == d:
        image = image[0, :, :].cpu().numpy()

    c = None
    cmap = None

    if cpw_mode:
        cmap = 'viridis'
        c = image[0,:]
    else:
        c = 'k'

    if ax is not None:
        ax.scatter(image[0,:], image[1,:], s=s, c=c, cmap=cmap)
    else:
        plt.scatter(image[0,:], image[1,:], s=s, c=c, cmap=cmap)

def wing_2D_shape_plot(wing, cpw_mode=False, s=0.5, axs=None, dpi=400):

    # Take first image of batch
    d = 4

    if len(wing.shape) == d:
        wing = wing[0, :, :].cpu().numpy()
    if isinstance(wing, torch.Tensor):
        wing = wing.cpu().numpy()

    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(20, 5), dpi=dpi)

    c = None
    cmap = None

    if cpw_mode:
        cmap = 'viridis'
        c = wing[0,0,:]
        x1 = wing[0,1,:]
        y1 = wing[0,2,:]
        x2 = wing[-1,1,:]
        y2 = wing[-1,2,:]
    else:
        c = 'k'
        x1 = wing[0,0,:]
        y1 = wing[0,1,:]
        x2 = wing[-1,0,:]
        y2 = wing[-1,1,:]

    axs[0].scatter(x1, y1, s=s, c=c, cmap=cmap)
    axs[1].scatter(x2, y2, s=s, c=c, cmap=cmap)
    # Remove x and y axes, and ticks
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    # axs[1].set_xticks([])
    # axs[1].set_yticks([])
    # Remove the top and right spines
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].spines['left'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    # Set the font size
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[1].tick_params(axis='both', which='major', labelsize=14)

def wing_3D_shape_plot(wing, cpw_mode=False, wing_len=2.25, ax=None, alpha=1.0, facecolor='steelblue', dpi=400, z=None, slice_mode=False):
    """
    Plot 3D wing shape with proper separation of upper and lower surfaces.

    Parameters:
    -----------
    wing : numpy.ndarray or torch.Tensor
        Airfoil slices data. Shape can be:
        - (n_slices, 3, n_points) if cpw_mode=True (x, y, color)
        - (n_slices, 2, n_points) if cpw_mode=False (x, y)
    wing_len : float
        Length of the wing (span)
    ax : matplotlib 3D axis, optional
        If None, creates new figure and axis
    alpha : float
        Transparency for the surfaces
    facecolor : str
        Color for the wing surface
    dpi : int
        DPI for the figure
    z : array-like, optional
        Custom z positions for each slice
    slice_mode : bool
        If True, plot only slice lines without surfaces
    """
    # Convert torch tensor to numpy if needed
    if hasattr(wing, 'cpu'):
        wing = wing.cpu().numpy()

    # Handle different data formats
    if len(wing.shape) == 4:
        wing = wing[0]  # Take first batch

    # Determine number of slices and points
    n_slices = wing.shape[0]

    if cpw_mode:
        # Format: (n_slices, 3, n_points) where 3 = [x, y, color]
        n_points = wing.shape[2]
        x_coords = wing[:, 0, :]
        y_coords = wing[:, 1, :]
    else:
        # Format: (n_slices, 2, n_points) or (n_slices, n_points, 2)
        if len(wing.shape) == 3:
            if wing.shape[1] == 2:
                # Format: (n_slices, 2, n_points)
                n_points = wing.shape[2]
                x_coords = wing[:, 0, :]
                y_coords = wing[:, 1, :]
            else:
                # Format: (n_slices, n_points, 2)
                n_points = wing.shape[1]
                x_coords = wing[:, :, 0]
                y_coords = wing[:, :, 1]
        else:
            raise ValueError(f"Unexpected wing shape: {wing.shape}")

    # Create figure and axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(20, 5), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')

    # Create z coordinates (spanwise positions)
    if z is None:
        z_positions = np.linspace(0, wing_len, n_slices)
    else:
        z_positions = z

    # For each slice, separate upper and lower surfaces
    # The airfoil points should be ordered from trailing edge, around leading edge, back to trailing edge
    # We need to find the leading edge (minimum x) to split the airfoil

    # Collect upper and lower surfaces for each slice
    upper_surfaces_x = []
    upper_surfaces_y = []
    lower_surfaces_x = []
    lower_surfaces_y = []

    for s in range(n_slices):
        x = x_coords[s]
        y = y_coords[s]

        # Find leading edge (minimum x) and trailing edge (maximum x)
        le_idx = np.argmin(x)
        te_idx = np.argmax(x)

        # Order points from leading edge to trailing edge for upper surface
        # Upper surface: from leading edge to trailing edge (going forward)
        if le_idx < te_idx:
            upper_indices = np.arange(le_idx, te_idx + 1)
            # Lower surface: from trailing edge to leading edge (wrapping around)
            lower_indices = np.concatenate([np.arange(te_idx, n_points), np.arange(0, le_idx + 1)])
        else:
            # Leading edge is after trailing edge (wrap around)
            upper_indices = np.concatenate([np.arange(le_idx, n_points), np.arange(0, te_idx + 1)])
            lower_indices = np.arange(te_idx, le_idx + 1)

        # Store the surfaces (we'll reverse lower surface to maintain consistent orientation)
        upper_surfaces_x.append(x[upper_indices])
        upper_surfaces_y.append(y[upper_indices])
        lower_surfaces_x.append(x[lower_indices[::-1]])  # Reverse to go from leading to trailing edge
        lower_surfaces_y.append(y[lower_indices[::-1]])

    # Plot slice lines if slice_mode is True
    if slice_mode:
        for s in range(n_slices):
            # Plot upper surface
            ax.plot(upper_surfaces_x[s], z_positions[s] * np.ones_like(upper_surfaces_x[s]),
                   upper_surfaces_y[s], '-', color=facecolor, alpha=alpha, linewidth=2.0)
            # Plot lower surface
            ax.plot(lower_surfaces_x[s], z_positions[s] * np.ones_like(lower_surfaces_x[s]),
                   lower_surfaces_y[s], '-', color=facecolor, alpha=alpha, linewidth=2.0)

            # Connect leading edge points
            ax.plot([upper_surfaces_x[s][0], lower_surfaces_x[s][0]],
                   [z_positions[s], z_positions[s]],
                   [upper_surfaces_y[s][0], lower_surfaces_y[s][0]],
                   '-', color=facecolor, alpha=alpha*0.5, linewidth=1.0)
            # Connect trailing edge points
            ax.plot([upper_surfaces_x[s][-1], lower_surfaces_x[s][-1]],
                   [z_positions[s], z_positions[s]],
                   [upper_surfaces_y[s][-1], lower_surfaces_y[s][-1]],
                   '-', color=facecolor, alpha=alpha*0.5, linewidth=1.0)

    else:
        # Create surface meshes by interpolating between slices
        # We need to ensure both upper and lower surfaces have the same number of points
        # for proper surface creation

        # Determine the maximum number of points among all slices for upper surface
        max_upper_points = max(len(u) for u in upper_surfaces_x)
        max_lower_points = max(len(l) for l in lower_surfaces_x)

        # Interpolate to have consistent point count
        from scipy.interpolate import interp1d

        # For upper surface
        X_upper_grid = np.zeros((n_slices, max_upper_points))
        Y_upper_grid = np.zeros((n_slices, max_upper_points))
        Z_upper_grid = np.zeros((n_slices, max_upper_points))

        for s in range(n_slices):
            n_pts = len(upper_surfaces_x[s])
            if n_pts < max_upper_points:
                # Interpolate to add points
                t_original = np.linspace(0, 1, n_pts)
                t_new = np.linspace(0, 1, max_upper_points)
                f_x = interp1d(t_original, upper_surfaces_x[s], kind='linear', fill_value='extrapolate')
                f_y = interp1d(t_original, upper_surfaces_y[s], kind='linear', fill_value='extrapolate')
                X_upper_grid[s] = f_x(t_new)
                Y_upper_grid[s] = f_y(t_new)
            else:
                X_upper_grid[s] = upper_surfaces_x[s][:max_upper_points]
                Y_upper_grid[s] = upper_surfaces_y[s][:max_upper_points]
            Z_upper_grid[s] = z_positions[s]

        # For lower surface
        X_lower_grid = np.zeros((n_slices, max_lower_points))
        Y_lower_grid = np.zeros((n_slices, max_lower_points))
        Z_lower_grid = np.zeros((n_slices, max_lower_points))

        for s in range(n_slices):
            n_pts = len(lower_surfaces_x[s])
            if n_pts < max_lower_points:
                # Interpolate to add points
                t_original = np.linspace(0, 1, n_pts)
                t_new = np.linspace(0, 1, max_lower_points)
                f_x = interp1d(t_original, lower_surfaces_x[s], kind='linear', fill_value='extrapolate')
                f_y = interp1d(t_original, lower_surfaces_y[s], kind='linear', fill_value='extrapolate')
                X_lower_grid[s] = f_x(t_new)
                Y_lower_grid[s] = f_y(t_new)
            else:
                X_lower_grid[s] = lower_surfaces_x[s][:max_lower_points]
                Y_lower_grid[s] = lower_surfaces_y[s][:max_lower_points]
            Z_lower_grid[s] = z_positions[s]

        # Plot upper surface as a mesh
        for i in range(n_slices - 1):
            for j in range(max_upper_points - 1):
                # Create quadrilateral for upper surface
                verts = [
                    [X_upper_grid[i, j], Z_upper_grid[i, j], Y_upper_grid[i, j]],
                    [X_upper_grid[i, j+1], Z_upper_grid[i, j+1], Y_upper_grid[i, j+1]],
                    [X_upper_grid[i+1, j+1], Z_upper_grid[i+1, j+1], Y_upper_grid[i+1, j+1]],
                    [X_upper_grid[i+1, j], Z_upper_grid[i+1, j], Y_upper_grid[i+1, j]]
                ]
                ax.add_collection3d(Poly3DCollection([verts], facecolor=facecolor, alpha=alpha, edgecolor='none'))

        # Plot lower surface as a mesh
        for i in range(n_slices - 1):
            for j in range(max_lower_points - 1):
                # Create quadrilateral for lower surface
                verts = [
                    [X_lower_grid[i, j], Z_lower_grid[i, j], Y_lower_grid[i, j]],
                    [X_lower_grid[i, j+1], Z_lower_grid[i, j+1], Y_lower_grid[i, j+1]],
                    [X_lower_grid[i+1, j+1], Z_lower_grid[i+1, j+1], Y_lower_grid[i+1, j+1]],
                    [X_lower_grid[i+1, j], Z_lower_grid[i+1, j], Y_lower_grid[i+1, j]]
                ]
                ax.add_collection3d(Poly3DCollection([verts], facecolor=facecolor, alpha=alpha*0.9, edgecolor='none'))

    # Set labels and limits
    ax.set_xlabel('x')
    ax.set_ylabel('z (span)')
    ax.set_zlabel('y')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, wing_len)
    ax.set_zlim(-0.3, 0.3)

    # Set aspect ratio
    ax.set_box_aspect([1.2, 1, 0.6])

    return ax

def wing_3D_pressure_plot(wing, pressure, ax=None, wing_len=2.25, alpha=1.0,
                          cmap='RdBu_r', vmin=None, vmax=None, z=None):
    """Plot 3D wing surface colored by pressure.

    Parameters
    ----------
    wing     : [S, 2, 192]  (x, y coords per slice)
    pressure : [S, 192]     (scalar pressure per point per slice)
    """
    if hasattr(wing, 'cpu'):
        wing = wing.cpu().numpy()
    if hasattr(pressure, 'cpu'):
        pressure = pressure.cpu().numpy()

    n_slices, _, n_points = wing.shape
    x_coords = wing[:, 0, :]
    y_coords = wing[:, 1, :]

    if z is None:
        z_positions = np.linspace(0, wing_len, n_slices)
    else:
        z_positions = z

    # Split each slice into upper/lower, tracking the original point indices
    upper_x, upper_y, upper_p = [], [], []
    lower_x, lower_y, lower_p = [], [], []
    for s in range(n_slices):
        x, y, p = x_coords[s], y_coords[s], pressure[s]
        le_idx = int(np.argmin(x))
        te_idx = int(np.argmax(x))
        if le_idx < te_idx:
            ui = np.arange(le_idx, te_idx + 1)
            li = np.concatenate([np.arange(te_idx, n_points), np.arange(0, le_idx + 1)])
        else:
            ui = np.concatenate([np.arange(le_idx, n_points), np.arange(0, te_idx + 1)])
            li = np.arange(te_idx, le_idx + 1)
        upper_x.append(x[ui]); upper_y.append(y[ui]); upper_p.append(p[ui])
        lower_x.append(x[li[::-1]]); lower_y.append(y[li[::-1]]); lower_p.append(p[li[::-1]])

    # Interpolate to uniform point count across slices
    from scipy.interpolate import interp1d as _interp1d
    def _resample(arrs):
        n = max(len(a) for a in arrs)
        out = np.zeros((n_slices, n))
        for s, a in enumerate(arrs):
            if len(a) == n:
                out[s] = a
            else:
                t0 = np.linspace(0, 1, len(a))
                t1 = np.linspace(0, 1, n)
                out[s] = _interp1d(t0, a, kind='linear', fill_value='extrapolate')(t1)
        return out

    UX = _resample(upper_x); UY = _resample(upper_y); UP = _resample(upper_p)
    LX = _resample(lower_x); LY = _resample(lower_y); LP = _resample(lower_p)

    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection='3d')

    colormap = plt.get_cmap(cmap)
    all_p = np.concatenate([UP.ravel(), LP.ravel()])
    pmin = vmin if vmin is not None else float(np.nanpercentile(all_p, 2))
    pmax = vmax if vmax is not None else float(np.nanpercentile(all_p, 98))

    def _norm(v):
        return float(np.clip((v - pmin) / (pmax - pmin + 1e-12), 0, 1))

    def _add_surface(X, Y, P, a):
        ns, np_ = X.shape
        for i in range(ns - 1):
            for j in range(np_ - 1):
                face_p = _norm(0.25 * (P[i,j] + P[i,j+1] + P[i+1,j] + P[i+1,j+1]))
                color  = colormap(face_p)
                verts  = [
                    [X[i,j],   z_positions[i],   Y[i,j]],
                    [X[i,j+1], z_positions[i],   Y[i,j+1]],
                    [X[i+1,j+1], z_positions[i+1], Y[i+1,j+1]],
                    [X[i+1,j], z_positions[i+1], Y[i+1,j]],
                ]
                ax.add_collection3d(Poly3DCollection([verts], facecolor=color,
                                                     alpha=a, edgecolor='none'))

    _add_surface(UX, UY, UP, alpha)
    _add_surface(LX, LY, LP, alpha * 0.9)

    # Colorbar via a ScalarMappable
    import matplotlib.cm as _cm
    import matplotlib.colors as _colors
    sm = _cm.ScalarMappable(cmap=colormap,
                            norm=_colors.Normalize(vmin=pmin, vmax=pmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1, label='Cp')

    ax.set_xlabel('x'); ax.set_ylabel('z (span)'); ax.set_zlabel('y')
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(0, wing_len); ax.set_zlim(-0.3, 0.3)
    ax.set_box_aspect([1.2, 1, 0.6])
    return ax


def show_forward_process(image,forward_fcn,bae_model, alpha=None, forward_fcn_alpha=None, T_m=100, num_images=6, reverse_transform_fcn=None,
                         denormalize_output=True, xlim_enc=[-1.0,1.0], ylim_enc=[-1.0,1.0], xlim_dec=[-0.005, 1.005], ylim_dec=[-0.3,0.3], figsize=(20,3),
                         alpha_mean_std=None,device='cpu'):
    """
    Simulates and plots the forward diffusion process for a given image.
    """
    # Args:
    #     image: Image to be diffused
    #     forward_fcn: Forward diffusion sampler with input (image, t)
    #     bae_model: Bezier Autoencoder model
    #     T_m: Number of timesteps to plot
    #     num_images: Number of images to plot
    #     reverse_transform_fcn: Function to reverse scaling/shifting transformation of the latent space.

    # Simulate and plot forward diffusion process            
    T_vec = np.linspace(0, T_m-1, num_images)
    T_vec = T_vec.astype(int)
    # Create a axis and figure
    fig, ax = plt.subplots(2, num_images, figsize=figsize)
    bae_model.to(device)

    for i, idx in enumerate(T_vec):
        t = torch.Tensor([idx]).type(torch.int64)
        # ax[0, i].axis('off')

        image_plt, noise = forward_fcn(image, t)
        # show_tensor_image(image_plt, noscale=True, cpw_mode=True, s= 4.0, ax=ax[0, i])

        if alpha is not None:
            alpha_plt, alpha_noise = forward_fcn_alpha(alpha, t)
            alpha_plt = alpha_plt.cpu().detach().numpy()
        if reverse_transform_fcn is not None:
            image_plt = reverse_transform_fcn(image_plt)

        # # Print min and max of the image
        # print(f"Min x: {torch.min(image_plt[:,1])}, Max x: {torch.max(image_plt[:,1])}")
        # print(f"Min y: {torch.min(image_plt[:,2])}, Max y: {torch.max(image_plt[:,2])}")

        show_tensor_image(image_plt, noscale=True, cpw_mode=True, s= 4.0, ax=ax[0, i])
        # ax[0, i].set_xlim(xlim_enc)
        # ax[0, i].set_ylim(ylim_enc)
        # Set the ylim differently for the first image
        if idx == 0:
            # Add a colorbar
            plt.colorbar(ax[0, i].collections[0], ax=ax[0, i],location='left')
        if alpha is not None:
            if alpha_plt.shape[0] > 1:
                alpha_plt = np.mean(alpha_plt)
            
            if alpha_mean_std is not None:
                alpha_plt = alpha_plt*alpha_mean_std[1] + alpha_mean_std[0]

            ax[0, i].set_title(r"t="+str(idx)+r", $\alpha$="+str(np.round(alpha_plt[0][0],2)))
            ax[0, i].axis('off')
            ax[0, i].axis('equal')
        else:
            ax[0, i].set_title(r"t="+str(idx))
            ax[0, i].axis('off')
            ax[0, i].axis('equal')
        
        # Set axis equal
        ax[1, i].axis('equal')

        ax[1, i].axis('off')
        decoded_airfoil_noised = bae_model.decode_z(image_plt, z_ae_mode=True, normalized_data=False, denormalize_output=denormalize_output)[0]
        ax[1, i].plot(decoded_airfoil_noised[0,0,:].cpu().detach().numpy(), decoded_airfoil_noised[0,1,:].cpu().detach().numpy(), 'k.')
        ax[1, i].set_xlim(xlim_dec)
        ax[1, i].set_ylim(ylim_dec)
    
    # Show the figure
    plt.show()

def show_forward_process_3D(
        image,
        alpha, 
        sampler,
        bae_model, 
        T_m=100, 
        num_images=6, 
        xlim_dec=[-0.005, 1.005], 
        ylim_dec=[-0.3,0.3], 
        figsize=(20,3),
        alpha_mean_std=None,
        device='cpu',
        dpi =400,
        shape_3D_plot=True,
        ):
    """
    Simulates and plots the forward diffusion process for a given image.
    """
    # Args:
    #     image: Image to be diffused
    #     forward_fcn: Forward diffusion sampler with input (image, t)
    #     bae_model: Bezier Autoencoder model
    #     T_m: Number of timesteps to plot
    #     num_images: Number of images to plot
    #     reverse_transform_fcn: Function to reverse scaling/shifting transformation of the latent space.

    # Simulate and plot forward diffusion process            
    T_vec = np.linspace(0, T_m-1, num_images)
    T_vec = T_vec.astype(int)
    # Create a axis and figure
    fig, ax = plt.subplots(2, num_images, figsize=figsize, dpi=dpi, subplot_kw={'projection': '3d'})

    bae_model.to(device)

    forward_fcn_x = sampler.schedule_x.forward_diffusion_sample
    forward_fcn_AoA = sampler.schedule_AoA.forward_diffusion_sample

    for i, idx in enumerate(T_vec):
        t = torch.Tensor([idx]).type(torch.int64)
        # ax[0, i].axis('off')

        # image_plt, noise = forward_fcn(image, t)
        image_plt, noise = forward_fcn_x(image, t, device)
        # show_tensor_image(image_plt, noscale=True, cpw_mode=True, s= 4.0, ax=ax[0, i])

        if alpha is not None:
            # alpha_plt, alpha_noise = forward_fcn_alpha(alpha, t)
            alpha_plt, alpha_noise = forward_fcn_AoA(alpha, t)
            alpha_plt = alpha_plt.cpu().detach().numpy()

        if shape_3D_plot:
            wing_3D_shape_plot(image_plt[0], cpw_mode=True, ax=ax[0, i], alpha=1.0, dpi=dpi)
        else:
            pass

        # ax[0, i].set_xlim(xlim_enc)
        # ax[0, i].set_ylim(ylim_enc)
        # Set the ylim differently for the first image
        if idx == 0:
            # Add a colorbar
            plt.colorbar(ax[0, i].collections[0], ax=ax[0, i],location='left')
        if alpha is not None:
            if alpha_plt.shape[0] > 1:
                alpha_plt = np.mean(alpha_plt)
            
            if alpha_mean_std is not None:
                alpha_plt = alpha_plt*alpha_mean_std[1] + alpha_mean_std[0]

            ax[0, i].set_title(r"t="+str(idx)+r", $\alpha$="+str(np.round(alpha_plt[0][0],2)))
            ax[0, i].axis('off')
            ax[0, i].axis('equal')
        else:
            ax[0, i].set_title(r"t="+str(idx))
            ax[0, i].axis('off')
            ax[0, i].axis('equal')
        
        # Set axis equal
        ax[1, i].axis('equal')

        ax[1, i].axis('off')
        decoded_airfoil_noised = bae_model.decode_z(image_plt[0], z_ae_mode=True, normalized_data=False, denormalize_output=True)[0]
        wing_3D_shape_plot(decoded_airfoil_noised.cpu().numpy(), cpw_mode=False, ax=ax[1, i], alpha=1.0, dpi=dpi)
    
    # Show the figure
    plt.show()

def show_forward_process_AoA(
        image,
        alpha, 
        sampler,
        bae_model, 
        T_m=100, 
        num_images=6, 
        xlim_dec=[-0.005, 1.005], 
        ylim_dec=[-0.3,0.3], 
        figsize=(20,3),
        alpha_mean_std=None,
        device='cpu'
        ):
    """
    Simulates and plots the forward diffusion process for a given image.
    """
    # Args:
    #     image: Image to be diffused
    #     forward_fcn: Forward diffusion sampler with input (image, t)
    #     bae_model: Bezier Autoencoder model
    #     T_m: Number of timesteps to plot
    #     num_images: Number of images to plot
    #     reverse_transform_fcn: Function to reverse scaling/shifting transformation of the latent space.

    forward_fcn_x = sampler.schedule_x.forward_diffusion_sample
    forward_fcn_AoA = sampler.schedule_AoA.forward_diffusion_sample

    # Simulate and plot forward diffusion process            
    T_vec = np.linspace(0, T_m-1, num_images)
    T_vec = T_vec.astype(int)
    # Create a axis and figure
    fig, ax = plt.subplots(2, num_images, figsize=figsize)
    bae_model.to(device)

    for i, idx in enumerate(T_vec):
        t = torch.Tensor([idx]).type(torch.int64).to(device)
        # ax[0, i].axis('off')

        image_plt, noise = forward_fcn_x(image, t, device)
        # show_tensor_image(image_plt, noscale=True, cpw_mode=True, s= 4.0, ax=ax[0, i])

        if alpha is not None:
            alpha_plt, alpha_noise = forward_fcn_AoA(alpha, t, device)
            alpha_plt = alpha_plt.cpu().detach().numpy()

        show_tensor_image(image_plt, noscale=True, cpw_mode=True, s= 4.0, ax=ax[0, i])
        # ax[0, i].set_xlim(xlim_enc)
        # ax[0, i].set_ylim(ylim_enc)
        # Set the ylim differently for the first image
        if idx == 0:
            # Add a colorbar
            plt.colorbar(ax[0, i].collections[0], ax=ax[0, i],location='left')
        if alpha is not None:
            if alpha_plt.shape[0] > 1:
                alpha_plt = np.mean(alpha_plt)
            
            if alpha_mean_std is not None:
                alpha_plt = alpha_plt*alpha_mean_std[1] + alpha_mean_std[0]

            ax[0, i].set_title(r"t="+str(idx)+r", $\alpha$="+str(np.round(alpha_plt[0][0],2)))
            ax[0, i].axis('off')
            ax[0, i].axis('equal')
        else:
            ax[0, i].set_title(r"t="+str(idx))
            ax[0, i].axis('off')
            ax[0, i].axis('equal')
        
        # Set axis equal
        ax[1, i].axis('equal')

        ax[1, i].axis('off')
        decoded_airfoil_noised = bae_model.decode_z(image_plt, z_ae_mode=True, normalized_data=False, denormalize_output=True)[0]
        ax[1, i].plot(decoded_airfoil_noised[0,0,:].cpu().detach().numpy(), decoded_airfoil_noised[0,1,:].cpu().detach().numpy(), 'k.')
        ax[1, i].set_xlim(xlim_dec)
        ax[1, i].set_ylim(ylim_dec)
    
    # Show the figure
    plt.show()

def sample_plot_image_3D(model, epoch, sample_timestep_fcn, bae_model,
                      num_images=6, T=100, animation_mode=False, dims=(1,9,3,30),
                      save=True, save_path=os.path.join('..', 'results','videos','diffusion_training'),
                      c=None, device='cpu', seed=None, figsize=(20,3), s = 2.0, alpha_mean_std=None, 
                      airfoil_init=None, xlim_dec=[-0.005, 1.005], ylim_dec=[-0.3,0.3],
                      dpi=300
                      ):
    """
    Plots a sample image from the diffusion model.
    Args:
        model: Diffusion model
        epoch: Epoch of the model
        sample_timestep_fcn: Function to sample a timestep from the diffusion model
        bae_model: Bezier Autoencoder model
        dims: Dimensions of the image
        num_images: Number of images to plot
        T: Number of timesteps
        animation_mode: If True, saves the image for animation instead of plotting
        noscale: If True, does not scale the image
        save: If True, saves the image
        save_path: Path to save the image
        c: Conditional input to the diffusion model
        reverse_transform: Function to reverse scaling/shifting transformation of the latent space.
        device: Device to run the model on
    """
    with torch.no_grad():
        if seed is not None:
            torch.manual_seed(seed)
        
        # image = torch.clone(airfoil_init)
        image = torch.randn(dims, device=device)
        print
        alpha = torch.randn(dims[0], 1, device=device)
        T_vec_full = np.linspace(0, T-1, T)
        
        T_vec = np.linspace(0, T-1, num_images)
        T_vec = T_vec.astype(int)
        # Reverse the order of the T_vec
        T_vec = T_vec[::-1]
        T_vec_full = T_vec_full[::-1]
        bae_model.to(device)

        if not animation_mode:
            # Create a axis and figure
            fig, ax = plt.subplots(2, num_images, figsize=figsize, dpi=dpi, subplot_kw={'projection': '3d'})

        for i, idx in enumerate(T_vec_full):
            t = torch.full((1,), idx, device=device, dtype=torch.long)
            image, alpha = sample_timestep_fcn(model, image, alpha, c, airfoil_init, t, t_mask=None)
            if np.any(T_vec == idx):
                plot_id = np.where(T_vec == idx)[0][0]
                if not animation_mode:
                    ax[0, plot_id].axis('off')

                image_plt = image.detach().clone().cpu()
                alpha_plt = alpha.detach().clone().cpu().numpy()

                if alpha_plt.shape[0] > 1:
                    alpha_plt = np.mean(alpha_plt)

                if alpha_mean_std is not None:
                    alpha_plt = alpha_plt*alpha_mean_std[1] + alpha_mean_std[0]

                if not animation_mode:
                    wing_3D_shape_plot(image_plt, cpw_mode=True, ax=ax[0, plot_id])
                    ax[0, plot_id].set_title(r"t="+str(int(idx))+r", $\alpha$="+str(np.round(alpha_plt[0][0],2)))
                    ax[0, plot_id].axis('off')
                    ax[0, plot_id].axis('equal')
                    # Add a colorbar
                    if plot_id == 0:
                        plt.colorbar(ax[0, plot_id].collections[0], ax=ax[0, plot_id],location='left')
                else:
                    plt.figure(figsize=figsize)
                    plt.axis('off')
                    plt.axis('equal')
                    plt.title(r"t="+str(int(idx))+r", $\alpha$="+str(np.round(alpha_plt[0][0],2)))
                    wing_3D_shape_plot(image_plt, cpw_mode=True, s=s, ax=ax[0, plot_id])
                    # set dpi to 1000 for high quality
                    plt.savefig(os.path.join(save_path, f'latent_space_{i}.png'), dpi=dpi, bbox_inches='tight')
                    # # Add a colorbar
                    # if plot_id == 0:
                    #     plt.colorbar(ax[0, plot_id].collections[0], ax=ax[0, plot_id],location='left')

                    # clear the figure
                    plt.clf()
                decoded_airfoil_noised = bae_model.decode_z(image_plt[0], z_ae_mode=True, denormalize_output=True, normalized_data=False)[0]

                if not animation_mode:
                    ax[1, plot_id].axis('equal')
                    ax[1, plot_id].axis('off')
                    wing_3D_shape_plot(decoded_airfoil_noised.cpu().numpy(), cpw_mode=False, ax=ax[1, plot_id], alpha=1.0, dpi=dpi)
                    
                    # plt.xlim(xlim_dec)
                    # plt.ylim(ylim_dec)
                    ax[1, plot_id].set_xlim(xlim_dec)
                    ax[1, plot_id].set_ylim(ylim_dec)
                
                else:
                    plt.figure(figsize=figsize)
                    wing_3D_shape_plot(decoded_airfoil_noised.cpu().numpy(), cpw_mode=False, ax=ax[1, plot_id], alpha=1.0, dpi=dpi)
                    plt.axis('equal')
                    plt.axis('off')
                    plt.xlim(xlim_dec)
                    plt.ylim(ylim_dec)
                    plt.savefig(os.path.join(save_path, f'airfoil_{i}.png'),dpi=dpi, bbox_inches='tight')
                    plt.clf()

        # save image for animation
        if save and not animation_mode:
            plt.savefig(save_path + f'epoch_{epoch}.png')
        plt.show()  


def sample_plot_image(model, epoch, sample_timestep_fcn, bae_model,
                      num_images=6, T=100, animation_mode=False, dims=(1,3,30),
                      save=True, save_path=os.path.join('..', 'results','videos','diffusion_training'),
                      c=None, device='cpu', seed=None, figsize=(20,3), s = 2.0, alpha_mean_std=None, 
                      airfoil_init=None, xlim_dec=[-0.005, 1.005], ylim_dec=[-0.3,0.3],
                      dpi=800,
                      modeled=False
                      ):
    """
    Plots a sample image from the diffusion model.
    Args:
        model: Diffusion model
        epoch: Epoch of the model
        sample_timestep_fcn: Function to sample a timestep from the diffusion model
        bae_model: Bezier Autoencoder model
        dims: Dimensions of the image
        num_images: Number of images to plot
        T: Number of timesteps
        animation_mode: If True, saves the image for animation instead of plotting
        noscale: If True, does not scale the image
        save: If True, saves the image
        save_path: Path to save the image
        c: Conditional input to the diffusion model
        reverse_transform: Function to reverse scaling/shifting transformation of the latent space.
        device: Device to run the model on
    """
    with torch.no_grad():
        if seed is not None:
            torch.manual_seed(seed)
        
        # image = torch.clone(airfoil_init)
        # print(f"airfoil_init shape: {airfoil_init.shape}") 
        if modeled:
            image = airfoil_init.detach().cpu().to(device)
        else:
            image = torch.randn(dims, device=device)

        alpha = torch.randn(dims[0], 1, device=device)
        T_vec_full = np.linspace(0, T-1, T)
        
        T_vec = np.linspace(0, T-1, num_images)
        T_vec = T_vec.astype(int)
        # Reverse the order of the T_vec
        T_vec = T_vec[::-1]
        T_vec_full = T_vec_full[::-1]
        bae_model.to(device)

        if not animation_mode:
            # Create a axis and figure
            fig, ax = plt.subplots(2, num_images, figsize=figsize)

        for i, idx in enumerate(T_vec_full):
            t = torch.full((1,), idx, device=device, dtype=torch.long)
            image, alpha = sample_timestep_fcn(model, image, alpha, c, airfoil_init, t, t_mask=None)
            if np.any(T_vec == idx):
                plot_id = np.where(T_vec == idx)[0][0]
                if not animation_mode:
                    ax[0, plot_id].axis('off')

                image_plt = image.detach().clone().cpu()
                alpha_plt = alpha.detach().clone().cpu().numpy()

                if alpha_plt.shape[0] > 1:
                    alpha_plt = np.mean(alpha_plt)

                if alpha_mean_std is not None:
                    alpha_plt = alpha_plt*alpha_mean_std[1] + alpha_mean_std[0]

                if not animation_mode:
                    show_tensor_image(image_plt, noscale=True, cpw_mode=True, s=s, ax=ax[0, plot_id])
                    ax[0, plot_id].set_title(r"t="+str(int(idx))+r", $\alpha$="+str(np.round(alpha_plt[0][0],2)))
                    ax[0, plot_id].axis('off')
                    ax[0, plot_id].axis('equal')
                    # Add a colorbar
                    if plot_id == 0:
                        plt.colorbar(ax[0, plot_id].collections[0], ax=ax[0, plot_id],location='left')
                else:
                    plt.figure(figsize=figsize)
                    plt.axis('off')
                    plt.axis('equal')
                    plt.title(r"t="+str(int(idx))+r", $\alpha$="+str(np.round(alpha_plt[0][0],2)))
                    show_tensor_image(image_plt, noscale=True, cpw_mode=True, s=s)
                    # set dpi to 1000 for high quality
                    plt.savefig(os.path.join(save_path, f'latent_space_{i}.png'), dpi=dpi, bbox_inches='tight')
                    # # Add a colorbar
                    # if plot_id == 0:
                    #     plt.colorbar(ax[0, plot_id].collections[0], ax=ax[0, plot_id],location='left')

                    # clear the figure
                    plt.clf()
                decoded_airfoil_noised = bae_model.decode_z(image_plt, z_ae_mode=True, denormalize_output=True, normalized_data=False)[0]

                if not animation_mode:
                    ax[1, plot_id].axis('equal')
                    ax[1, plot_id].axis('off')
                    ax[1, plot_id].plot(decoded_airfoil_noised[0,0,:].cpu().detach().numpy(), decoded_airfoil_noised[0,1,:].cpu().detach().numpy(), 'k.')
                    # plt.xlim(xlim_dec)
                    # plt.ylim(ylim_dec)
                    ax[1, plot_id].set_xlim(xlim_dec)
                    ax[1, plot_id].set_ylim(ylim_dec)
                
                else:
                    plt.figure(figsize=figsize)
                    plt.plot(decoded_airfoil_noised[0,0,:].cpu().detach().numpy(), decoded_airfoil_noised[0,1,:].cpu().detach().numpy(), 'k.')
                    plt.title(r"t="+str(idx)+r", $\alpha$="+str(np.round(alpha_plt[0][0],2)))
                    plt.axis('equal')
                    plt.axis('off')
                    plt.xlim(xlim_dec)
                    plt.ylim(ylim_dec)
                    plt.savefig(os.path.join(save_path, f'airfoil_{i}.png'),dpi=dpi, bbox_inches='tight')
                    plt.clf()

        # save image for animation
        if save and not animation_mode:
            plt.savefig(save_path + f'epoch_{epoch}.png')
        plt.show()  

def sample_airfoil(model, sample_timestep_fcn, bae_model,
                      dims=(1,3,62), T=100,
                      c=None, t_mask=None, airfoil_init=None, airfoil_traj=None, aoa_traj=None, cfg_scale=None,
                      reverse_transform=None, alpha_mean_std=None, device='cpu', seed=None, cfg_conds=None, special=False
                      ):
    """
    Returns a sample generated airfoil and its latent coordinates 
    Args:
        model: Diffusion model
        sample_timestep_fcn: Function to sample a timestep from the diffusion model
        bae_model: Bezier Autoencoder model
        dims: Dimensions of the image
        T: Number of timesteps
        noscale: If True, does not scale the image
        c: Conditional input to the diffusion model
        reverse_transform: Function to reverse scaling/shifting transformation of the latent space.
        device: Device to run the model on
    """
    with torch.no_grad():
        if seed is not None:
            torch.manual_seed(seed)
        if not special:
            airfoil_latent = torch.randn(dims, device=device)
        else:
            airfoil_latent = torch.clone(airfoil_init)
        alpha_latent = torch.randn(dims[0], 1, device=device) 

        for i in range(0,T)[::-1]:
            t = torch.full((dims[0],), i, device=device, dtype=torch.long)
            airfoil_latent, alpha_latent = sample_timestep_fcn(model, airfoil_latent, t, alpha=alpha_latent, c=c, 
                                                               t_mask=t_mask, airfoil_init=airfoil_init, airfoil_traj=airfoil_traj, 
                                                               aoa_traj=aoa_traj, cfg_scale=cfg_scale, cfg_conds=cfg_conds)

        airfoil_latent = airfoil_latent.detach().clone()
        alpha_latent = alpha_latent.detach().clone()
        if reverse_transform is not None:
            airfoil_latent = reverse_transform(airfoil_latent)

        if alpha_mean_std is not None:
            gen_alpha = alpha_latent*alpha_mean_std[1] + alpha_mean_std[0]
        
        gen_airfoil = bae_model.decode_z(airfoil_latent, z_ae_mode=True, denormalize_output=True,normalized_data=False)[0]

        return gen_airfoil.cpu().detach().numpy(), gen_alpha.cpu().detach().numpy(), airfoil_latent.cpu().detach().numpy()

def tensor_to_list(tensor: torch.Tensor):
    """
    Converts a tensor to a list of tensors from the first dimension.
    """
    return [tensor[i].unsqueeze(0).cpu().detach().numpy() for i in range(tensor.shape[0])]

def convert_to_numpy_list(tensor_list):
    """
    Converts a list of tensors to a list of numpy arrays.
    """
    return [tensor.cpu().detach().numpy() for tensor in tensor_list]

def check_input_list(input_list):
    """
    Checks if the input list is a list of tensors or a tensor.
    """
    if isinstance(input_list, torch.Tensor):
        input_list = tensor_to_list(input_list)
        return input_list
    elif isinstance(input_list[0], torch.Tensor):
        input_list = convert_to_numpy_list(input_list)
        return input_list
    else:
        return input_list

def plot_generations(decoded_airfoil_list, alpha_list, 
                     latent_mode = False, c_list=None, figsize=(14,7), 
                     linewidth=2, x_lim=(-0.015,1.05), y_lim=(-0.125,0.125),
                     nrows=1, fontsize=30, only_alpha=False, 
                     save=False, savepath=None,
                     plot_title=None, 
                     ftypes=None,
                     gt_airfoil_list=None, gt_alpha_list=None,
                     leg_loc=[0.0,-0.30],
                     color='k',
                     add_axes=False,
                     axes_labels=['M', 'Re'],
                     params=[0,1]
                     ):
    """
    Plots the generated airfoils.
    Args:
        @decoded_airfoil_list: List of decoded airfoils
        @latent_mode: If True, plots the latent space
        @c_list: List of conditional inputs
        @figsize: Size of the figure
    """

    # If decoded_airfoil_list, or alpha_list is a tensor, convert it to a list of tensors from the first dimension
    decoded_airfoil_list = check_input_list(decoded_airfoil_list)
    alpha_list = check_input_list(alpha_list)

    if gt_airfoil_list is not None:
        gt_airfoil_list = check_input_list(gt_airfoil_list)
        MSE_airfoils = [np.mean((decoded_airfoil_list[i] - gt_airfoil_list[i])**2) for i in range(len(decoded_airfoil_list))]
    
    # if gt_alpha_list is not None:
    #     gt_alpha_list = check_input_list(gt_alpha_list)
    #     MSE_alpha = [np.mean((alpha_list[i] - gt_alpha_list[i])**2) for i in range(len(alpha_list))]

    if c_list is not None:
        c_list = check_input_list(c_list)

    plt.figure(figsize=figsize)
    # Create a plot title
    if plot_title is not None:
        plt.suptitle(plot_title, fontsize=fontsize, y=1.1)
    plt.axis('off')
    # figure out number of columns based on number of airfoils and number of rows
    ncols = int(np.ceil(len(decoded_airfoil_list)/nrows))

    for i in range(len(decoded_airfoil_list)):
        plt.subplot(nrows, ncols, i+1)
        plt.axis('off')
        if latent_mode:
            show_tensor_image(decoded_airfoil_list[i], noscale=True, cpw_mode=True)
        else:
            plt.plot(decoded_airfoil_list[i][0,0,:], decoded_airfoil_list[i][0,1,:], color, linewidth=linewidth)
            if gt_airfoil_list is not None:
                plt.plot(gt_airfoil_list[i][0,0,:], gt_airfoil_list[i][0,1,:], 'royalblue', linewidth=linewidth, alpha=0.7)

            # get rid of vertical white space
            # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.2, hspace=0)
            if x_lim is not None:
                plt.xlim(x_lim)
            if y_lim is not None:
                plt.ylim(y_lim)
        
        alpha_str = str(round(alpha_list[i][0][0], 2))
        if only_alpha:
            title = r'$\alpha$ = $'+alpha_str+r'^{\circ}$'
            if gt_airfoil_list is not None:
                # with scientific notation
                MSE_airfoil_str = "{:.1e}".format(MSE_airfoils[i])
                title += r', $L_{x} = $ '+MSE_airfoil_str
            # if gt_alpha_list is not None:
            #     MSE_alpha_str = "{:.1e}".format(MSE_alpha[i])
            #     title += r'$, L_{\alpha} = $'+MSE_alpha_str
            plt.title(title, fontsize=fontsize, loc='center')
        elif c_list is not None:
            # round to 2 decimal places
            Re = round((c_list[i][0,1])*10**(-6), 2)
            plt.title(r'$\alpha$ = $'+alpha_str+r'^{\circ}$'+f', $M$ = {c_list[i][0,0]:.1f}, \n $Re$ = {Re:.1f}E6, $C_L$ = {c_list[i][0,2]:.1f}'.format(), fontsize=fontsize)
        elif gt_airfoil_list is not None:
            if gt_airfoil_list is not None:
                # with scientific notation
                MSE_airfoil_str = "{:.1e}".format(MSE_airfoils[i])
                title = r'$L_{x} = $ '+ MSE_airfoil_str
                plt.title(title, fontsize=fontsize, loc='center')
        else:
            continue

        # Add legend to corner of plot if ground truth airfoils are provided
        # if gt_airfoil_list is not None and i == len(decoded_airfoil_list) - 1:
        #     plt.legend(['Generated', 'Ground Truth'], loc=leg_loc, fontsize=fontsize)
        if gt_airfoil_list is not None and i == ncols*(nrows - 1):
            plt.legend(['Generated', 'Ground Truth'], loc=leg_loc, fontsize=fontsize)
        
        # if add_axes and 


        if i == ncols*(nrows - 1) and add_axes:
            plt.axis('on')
            # Remove ticks
            plt.xticks([])
            plt.yticks([])
            # Remove upper and right spines
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.xlabel(axes_labels[0], fontsize=fontsize)
            plt.ylabel(axes_labels[1], fontsize=fontsize)
            # Add arrows to the axes
            plt.arrow(0.5, 0, 0, 0.5, head_width=0.05, head_length=0.05, fc='k', ec='k')
            plt.arrow(0, 0.5, 0.5, 0, head_width=0.05, head_length=0.05, fc='k', ec='k')
    
    # Remove some of the whitespace
    plt.tight_layout()
    if save:
        print(f"Saving plot to {savepath}")
        if ftypes is not None:
            for ftype in ftypes:
                plt.savefig(savepath + '.' + ftype, bbox_inches='tight')
        else:
            plt.savefig(savepath, bbox_inches='tight')
    plt.show()