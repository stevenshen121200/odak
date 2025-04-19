import sys
import argparse
import torch
import odak
from odak.learn.wave import multi_color_hologram_optimizer, multiplane_loss, propagator
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import os

__title__ = 'Multi-color Holograms'

import json
import numpy as np
def update_settings(save,loss_weights,file_path='settings/holoeye.txt'):
    with open(file_path, 'r') as file:
        data = json.load(file)

    data["general"]["output directory"] = save
    data['general']['loss weights'] = loss_weights

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def main():
    settings_filename = './settings/holoeye.txt'
    for l1 in [1]:
        for l2 in np.logspace(-4, 1, 50, base=10):
            ws = [l1,0,l2]
            update_settings(f'./ratios/mse{l1}tv{l2}/',ws)
            parser = argparse.ArgumentParser(description=__title__)
            parser.add_argument(
                '--settings',
                type=argparse.FileType('r'),
                help='Filename for the settings file. Default is {}'.format(settings_filename)
            )
            args = parser.parse_args()
            if type(args.settings) != type(None):
                settings_filename = str(args.settings.name)
            process(settings_fn=settings_filename)

def main_old():
    settings_filename = './settings/holoeye.txt'
    parser = argparse.ArgumentParser(description=__title__)
    parser.add_argument(
        '--settings',
        type=argparse.FileType('r'),
        help='Filename for the settings file. Default is {}'.format(settings_filename)
    )
    args = parser.parse_args()
    if type(args.settings) != type(None):
        settings_filename = str(args.settings.name)
    process(settings_fn=settings_filename)


def rgb_to_xyz(rgb):
    """
    Convert an RGB image tensor (shape: [3, H, W]) to the XYZ color space.
    Assumes the RGB image is in linear space and normalized to [0, 1].
    Uses the standard sRGB-to-XYZ conversion matrix (D65 illuminant).
    """
    conversion_matrix = torch.tensor(
        [[0.412453, 0.357580, 0.180423],
         [0.212671, 0.715160, 0.072169],
         [0.019334, 0.119193, 0.950227]],
        device=rgb.device, dtype=rgb.dtype
    )
    C, H, W = rgb.shape
    # Reshape to [3, H*W] then transpose to [H*W, 3] for matrix multiplication.
    rgb_flat = rgb.view(3, -1).T  # shape: [H*W, 3]
    xyz_flat = torch.matmul(rgb_flat, conversion_matrix.T)  # shape: [H*W, 3]
    xyz = xyz_flat.T.view(3, H, W)  # shape: [3, H, W]
    return xyz


def spectral_to_xyz(spectral_img):
    """
    Convert a 3-channel spectral image (650nm, 550nm, 450nm) to CIEXYZ.
    Input shape: [3, H, W]
    """
    # LMS cone sensitivity approximations at [650, 550, 450] nm
    # Values roughly estimated from Stockman & Sharpe 2-deg cone fundamentals
    # These can be fine-tuned with actual color matching functions (CMFs)
    # Columns: wavelengths 650nm, 550nm, 450nm
    lms_sensitivities = torch.tensor([
        [0.1, 0.7, 0.0],  # L-cone (long, red)
        [0.0, 1.0, 0.0],  # M-cone (medium, green)
        [0.0, 0.1, 0.8],  # S-cone (short, blue)
    ], dtype=spectral_img.dtype, device=spectral_img.device)  # shape: [3 (LMS), 3 (wavelengths)]

    # Convert [3, H, W] to [H*W, 3]
    C, H, W = spectral_img.shape
    spectral_flat = spectral_img.view(C, -1).T  # shape: [H*W, 3]

    # Compute LMS responses: [H*W, 3 (LMS)]
    lms_flat = spectral_flat @ lms_sensitivities.T

    # LMS to XYZ conversion matrix (Hunt-Pointer-Estevez)
    lms_to_xyz = torch.tensor([
        [1.94735469, -1.41445123, 0.36476327],
        [0.68990272, 0.34832189, 0.00000000],
        [0.00000000, 0.00000000, 1.93485343]
    ], dtype=spectral_img.dtype, device=spectral_img.device)

    # Compute XYZ: [H*W, 3]
    xyz_flat = lms_flat @ lms_to_xyz.T

    # Reshape back to [3, H, W]
    xyz = xyz_flat.T.view(3, H, W)
    return xyz


def xyz_to_luv_luminance(xyz, white_point_Y=1.0):
    """
    Compute the L* (lightness) channel from an XYZ image using the CIE LUV formula.
    Here, xyz is a tensor of shape [3, H, W] and the Y channel is used.
    The standard piecewise function is applied:
      L* = 116*(Y/Yn)^(1/3) - 16  if Y/Yn > 0.008856, else L* = 903.3*(Y/Yn)
    The result is a single-channel image with values typically in [0, 100].
    """
    Y = xyz[1]  # Y channel
    Y_ratio = Y / white_point_Y
    L = torch.where(Y_ratio > 0.008856,
                    116.0 * torch.pow(Y_ratio, 1 / 3) - 16,
                    903.3 * Y_ratio)
    return L


def compansate_illumination(settings, target_image, device):
    illimuniation_form = odak.learn.tools.load_image(
        settings["beam"]["beam profile"],
        normalizeby=2 ** settings["target"]["color depth"],
        torch_style=True
    ).to(device)
    illimuniation_form_max = torch.amax(illimuniation_form, dim=(1, 2)).view(3, 1, 1)
    compensation = illimuniation_form / illimuniation_form_max
    target_image_com = target_image / compensation
    target_image_com = target_image_com / torch.amax(target_image_com, dim=(1, 2)).view(3, 1, 1)
    target_image = target_image_com * torch.amax(target_image, dim=(1, 2)).view(3, 1, 1)
    return target_image


def process(settings_fn):
    settings = odak.tools.load_dictionary(settings_fn)
    device = torch.device(settings['general']['device'])
    resolution = settings['spatial light modulator']['resolution']
    target_image = odak.learn.tools.load_image(
        settings["target"]["image filename"],
        normalizeby=2 ** settings["target"]["color depth"],
        torch_style=True
    ).to(device)[0:3, 0:resolution[0], 0:resolution[1]]
    target_depth = odak.learn.tools.load_image(
        settings["target"]["depth filename"],
        normalizeby=2 ** settings["target"]["color depth"],
        torch_style=True
    ).to(device)
    if len(target_depth.shape) > 2:
        target_depth = torch.mean(target_depth, dim=0)
    target_depth = target_depth[0:resolution[0], 0:resolution[1]]
    if settings["beam"]["beam profile"] != '':
        target_image = compansate_illumination(settings, target_image, device)
    loss_function = multiplane_loss(
        target_image=target_image,
        target_depth=target_depth,
        target_blur_size=settings["target"]["defocus blur size"],
        number_of_planes=settings["target"]["number of depth layers"],
        blur_ratio=settings["target"]["blur ratio"],
        weights=settings["target"]["weights"],
        scheme=settings["target"]["scheme"],
        reduction=settings['general']['reduction'],
        device=device
    )
    targets, focus_target, depth = loss_function.get_targets()
    propagator_mc = propagator(
        wavelengths=settings['beam']['wavelengths'],
        pixel_pitch=settings['spatial light modulator']['pixel pitch'],
        resolution=settings['spatial light modulator']['resolution'],
        aperture_size=settings['beam']['pinhole size'],
        number_of_frames=settings['target']['number of frames'],
        number_of_depth_layers=settings['target']['number of depth layers'],
        volume_depth=settings['target']['volume depth'],
        image_location_offset=settings['target']['location offset'],
        propagation_type=settings['beam']['propagation type'],
        propagator_type=settings['beam']['propagator type'],
        method=settings['general']['method'],
        device=device
    )
    mcho = multi_color_hologram_optimizer(
        wavelengths=settings["beam"]["wavelengths"],
        resolution=settings["spatial light modulator"]["resolution"],
        targets=targets,
        propagator=propagator_mc,
        number_of_frames=settings["target"]["number of frames"],
        number_of_depth_layers=settings['target']['number of depth layers'],
        learning_rate=settings["general"]["learning rate"],
        learning_rate_floor=settings["general"]["learning rate floor"],
        double_phase=settings["general"]["double phase constrain"],
        method=settings["general"]["method"],
        channel_power_filename=settings["target"]["channel power filename"],
        device=device,
        loss_function=loss_function,
        peak_amplitude=settings["target"]["peak amplitude"],
        optimize_peak_amplitude=settings["target"]["optimize peak amplitude"],
        img_loss_thres=settings["target"]["img loss threshold"],
        reduction=settings['general']['reduction'],
        number_of_slms=settings['spatial light modulator']['number of slms']
    )
    hologram_phases, frame_reconstructions, laser_powers, channel_powers, peak_amplitude = mcho.optimize(
        number_of_iterations=settings["general"]["iterations"],
        weights=settings["general"]["loss weights"]
    )
    settings['target']['peak amplitude'] = peak_amplitude
    save(
        settings,
        device,
        hologram_phases,
        laser_powers,
        channel_powers,
        frame_reconstructions,
        targets,
        target_image,
        target_depth,
        depth,
        settings['target']['peak amplitude'],
        settings['target']['color depth']
    )


def save(settings, device, hologram_phases, laser_powers, channel_powers, frame_reconstructions, targets, target_image,
         target_depth, depth, intensity_scale, color_depth):
    output_folder = settings["general"]["output directory"]
    directory = output_folder + settings["general"]["method"]
    odak.tools.check_directory(directory)
    odak.tools.save_dictionary(settings, '{}/settings.txt'.format(directory))
    checker_complex = odak.learn.wave.linear_grating(
        settings["spatial light modulator"]["resolution"][0],
        settings["spatial light modulator"]["resolution"][1],
        add=odak.pi,
        axis='y'
    ).to(device)
    checker = odak.learn.wave.calculate_phase(checker_complex)
    for depth_id in range(targets.shape[0]):
        odak.learn.tools.save_image(
            "{}/target_{:02d}.png".format(directory, depth_id), targets[depth_id] * intensity_scale,
            cmin=0.,
            cmax=intensity_scale,
            color_depth=color_depth
        )
        odak.learn.tools.save_image(
            "{}/reconstruction_{:02d}.png".format(directory, depth_id),
            torch.sum(frame_reconstructions[:, depth_id], dim=0),
            cmin=0.,
            cmax=intensity_scale,
            color_depth=color_depth
        )
    hologram_phases_w_grating = torch.zeros_like(hologram_phases)
    for frame_id in range(hologram_phases.shape[0]):  # Loop over frames
        for slm_id in range(hologram_phases.shape[1]):  # Loop over SLMs
            phase = hologram_phases[frame_id, slm_id]
            phase_normalized = phase % (2 * odak.pi)

            # Save phase for the current frame and SLM
            odak.learn.tools.save_image(
                "{}/phase_frame_{:02d}_slm_{:01d}.png".format(directory, frame_id, slm_id),
                phase_normalized,
                cmin=0.,
                cmax=odak.pi * 2
            )

            # Add checkerboard grating and normalize
            phase_grating = phase + checker
            phase_grating_normalized = phase_grating % (2 * odak.pi)
            hologram_phases_w_grating[frame_id, slm_id] = phase_grating_normalized

            # Save grating phase for the current frame and SLM
            odak.learn.tools.save_image(
                "{}/phase_grated_frame_{:02d}_slm_{:01d}.png".format(directory, frame_id, slm_id),
                phase_grating_normalized,
                cmin=0.,
                cmax=odak.pi * 2
            )

            for depth_id in range(targets.shape[0]):  # Loop over depths
                # Save reconstructions for each depth
                odak.learn.tools.save_image(
                    "{}/reconstruction_frame_{:02d}_depth_{:03d}.png".format(directory, frame_id,
                                                                             depth_id),
                    frame_reconstructions[frame_id, depth_id],
                    cmin=0.,
                    cmax=intensity_scale,
                    color_depth=color_depth
                )
    if hologram_phases.shape[0] == 3:
        for slm_id in range(hologram_phases.shape[1]):  # Loop over SLMs
            # Save combined phase images for the current SLM
            odak.learn.tools.save_image(
                '{}/phase_combined_slm_{:01d}.png'.format(directory, slm_id),
                hologram_phases[:, slm_id] % (2 * odak.pi),
                cmin=0.,
                cmax=odak.pi * 2.
            )
            # Save combined phase images with grating for the current SLM
            odak.learn.tools.save_image(
                '{}/phase_combined_w_grating_slm_{:01d}.png'.format(directory, slm_id),
                hologram_phases_w_grating[:, slm_id],
                cmin=0.,
                cmax=odak.pi * 2.
            )
    odak.learn.tools.save_torch_tensor('{}/laser_powers.pt'.format(directory), laser_powers)
    odak.learn.tools.save_torch_tensor('{}/channel_powers.pt'.format(directory), channel_powers)
    data = {
        "targets": targets,
        "target": target_image,
        "target depth": target_depth,
        "depth": depth,
        "intensity scale": intensity_scale,
        "laser powers": laser_powers,
        "channel powers": channel_powers,
        "hologram phases": hologram_phases,
        "settings": settings
    }
    odak.learn.tools.save_torch_tensor('{}/data.pt'.format(directory), data)

    # --- ADD THESE LINES BELOW to compute PSNR/SSIM ---
    # We assume both frame_reconstructions and targets have shape:
    #   frame_reconstructions: [num_frames, num_depths, C, H, W] or [num_frames, num_depths, H, W]
    #   targets:               [num_depths, C, H, W] or [num_depths, H, W]
    #
    # If your channel dimension is in a different place, you can adapt accordingly.
    # Below we assume channel-first convention: [C, H, W].

    print("\n=== PSNR and SSIM for each depth (summing over all frames) ===")
    for depth_id in range(targets.shape[0]):
        # Sum across all frames to get the final reconstruction for this depth
        # shape: [C, H, W]
        recon = torch.sum(frame_reconstructions[:, depth_id], dim=0)

        # Also ensure that target at this depth matches shape [C, H, W]
        # If your target is [C, H, W] already, you can use it directly
        target_for_depth = targets[depth_id]

        # Compute PSNR
        psnr_val = peak_signal_noise_ratio(
            recon.unsqueeze(0),  # shape => [1, C, H, W]
            target_for_depth.unsqueeze(0),  # shape => [1, C, H, W]
            data_range=intensity_scale  # Use intensity_scale if your data is scaled up to that
        )

        # Compute SSIM
        ssim_val = structural_similarity_index_measure(
            recon.unsqueeze(0),
            target_for_depth.unsqueeze(0),
            data_range=intensity_scale
        )
        message = f"Depth {depth_id:02d} -> PSNR: {psnr_val.item():.4f}, SSIM: {ssim_val.item():.4f}"
        print(message)

        file_path = os.path.join(directory, "psnr.txt")

        with open(file_path, "w") as f:
            f.write(message)

    print("\n=== PSNR in XYZ and PSNR in Lumanicity (CIE LUV) for each depth (summing over all frames) ===")
    for depth_id in range(targets.shape[0]):
        # Sum across all frames to get the final reconstruction for this depth (expected shape: [3, H, W])
        recon = torch.sum(frame_reconstructions[:, depth_id], dim=0)
        target_for_depth = targets[depth_id]  # expected shape: [3, H, W]

        # ----- PSNR in XYZ -----
        # Convert RGB images to the XYZ color space
        recon_xyz = spectral_to_xyz(recon)
        target_xyz = spectral_to_xyz(target_for_depth)

        # When using normalized sRGB (in [0, 1]), note that for pure white [1,1,1]:
        # X ≈ 0.950456, Y = 1.0, Z ≈ 1.088754. We choose the maximum (Z) as the data range.
        data_range_xyz = 1.088754
        psnr_xyz = peak_signal_noise_ratio(
            recon_xyz.unsqueeze(0),  # shape: [1, 3, H, W]
            target_xyz.unsqueeze(0),
            data_range=data_range_xyz
        )

        # ----- PSNR in Lumanicity (CIE LUV) -----
        # Compute the L* (lightness) channel from the XYZ representation.
        recon_L = xyz_to_luv_luminance(recon_xyz)  # shape: [H, W]
        target_L = xyz_to_luv_luminance(target_xyz)  # shape: [H, W]

        # L* values typically range from 0 to 100.
        data_range_L = 100.0
        # Reshape to [1, 1, H, W] for PSNR computation
        recon_L = recon_L.unsqueeze(0).unsqueeze(0)
        target_L = target_L.unsqueeze(0).unsqueeze(0)
        psnr_lum = peak_signal_noise_ratio(
            recon_L,
            target_L,
            data_range=data_range_L
        )
        print(f"Depth {depth_id:02d} -> PSNR_XYZ: {psnr_xyz.item():.4f}, PSNR_Lum (CIE LUV): {psnr_lum.item():.4f}")

    print('Output stored at {}. Check `odak.log` for more information.'.format(directory))


if __name__ == "__main__":
    sys.exit(main())
