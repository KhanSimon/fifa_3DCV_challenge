from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


from pathlib import Path
from typing import Union

from matplotlib.animation import FuncAnimation

# Connexions entre keypoints
SKELETON = [
    # Tête
    (0, 1),   # Nose -> Neck
    (0, 15),  # Nose -> REye
    (0, 16),  # Nose -> LEye
    (15, 17), # REye -> REar
    (16, 18), # LEye -> LEar

    # Bras droit
    (1, 2),   # Neck -> RShoulder
    (2, 3),   # RShoulder -> RElbow
    (3, 4),   # RElbow -> RWrist

    # Bras gauche
    (1, 5),   # Neck -> LShoulder
    (5, 6),   # LShoulder -> LElbow
    (6, 7),   # LElbow -> LWrist

    # Tronc / jambe droite
    (1, 8),   # Neck -> MidHip
    (8, 9),   # MidHip -> RHip
    (9, 10),  # RHip -> RKnee
    (10, 11), # RKnee -> RAnkle

    # Tronc / jambe gauche
    (8, 12),  # MidHip -> LHip
    (12, 13), # LHip -> LKnee
    (13, 14), # LKnee -> LAnkle

    # Pied gauche
    (14, 19), # LAnkle -> LBigToe
    (14, 20), # LAnkle -> LSmallToe
    (14, 21), # LAnkle -> LHeel

    # Pied droit
    (11, 22), # RAnkle -> RBigToe
    (11, 23), # RAnkle -> RSmallToe
    (11, 24), # RAnkle -> RHeel
]


def visualise_npy(file_path):
    
    # Load the .npy file
    data = np.load(file_path)

    # Check the shape of the data
    print(f"Data shape: {data.shape}")

    print(data)





def visualize_2D_npy(npy_path: Union[str, Path], index_player: int):
    
    """
    Charge un fichier .npy contenant des poses 2D de forme (T, N, 25, 2)
    et affiche une animation matplotlib des 25 keypoints d'un joueur donné,
    avec segments entre keypoints.

    Paramètres
    ----------
    npy_path : str | Path
        Chemin vers le fichier .npy.
    index_player : int
        Index du joueur à visualiser (dimension N).

    Retour
    ------
    anim : matplotlib.animation.FuncAnimation
        Objet animation matplotlib.
    """
    npy_path = Path(npy_path)
    data = np.load(npy_path)

    if data.ndim != 4:
        raise ValueError(
            f"Le tableau doit être de dimension 4 (T, N, 25, 2), reçu: {data.shape}"
        )

    T, N, K, D = data.shape

    if K != 25 or D != 2:
        raise ValueError(f"Shape attendue (_, _, 25, 2), reçu: {data.shape}")

    if not (0 <= index_player < N):
        raise IndexError(
            f"index_player={index_player} hors limites. "
            f"Nombre de joueurs disponible dans le tenseur: N={N}"
        )

    player_data = data[:, index_player, :, :]  # shape: (T, 25, 2)

    # Bornes globales sur toutes les frames valides du joueur
    x_all = player_data[..., 0]
    y_all = player_data[..., 1]

    valid_x = x_all[~np.isnan(x_all)]
    valid_y = y_all[~np.isnan(y_all)]

    if valid_x.size == 0 or valid_y.size == 0:
        raise ValueError(
            f"Le joueur {index_player} est absent de toutes les frames (que des NaN)."
        )

    x_min, x_max = valid_x.min(), valid_x.max()
    y_min, y_max = valid_y.min(), valid_y.max()

    margin_x = max((x_max - x_min) * 0.05, 1.0)
    margin_y = max((y_max - y_min) * 0.05, 1.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    scat = ax.scatter([], [], s=5)

    # Une line2D par segment
    lines = [ax.plot([], [], linewidth=2)[0] for _ in SKELETON]

    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_max + margin_y, y_min - margin_y)  # inversion axe Y image
    ax.set_title(f"Joueur {index_player} - frame 0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    def init():
        scat.set_offsets(np.empty((0, 2)))
        for line in lines:
            line.set_data([], [])
        return (scat, *lines)

    def update(frame_idx: int):
        points = player_data[frame_idx]  # (25, 2)

        # Keypoints valides
        valid_mask = ~np.isnan(points).any(axis=1)
        valid_points = points[valid_mask]

        if valid_points.size == 0:
            scat.set_offsets(np.empty((0, 2)))
            for line in lines:
                line.set_data([], [])
            ax.set_title(f"Joueur {index_player} - frame {frame_idx} (absent)")
            return (scat, *lines)

        scat.set_offsets(valid_points)

        # Mise à jour des segments
        for line, (i, j) in zip(lines, SKELETON):
            if valid_mask[i] and valid_mask[j]:
                x = [points[i, 0], points[j, 0]]
                y = [points[i, 1], points[j, 1]]
                line.set_data(x, y)
            else:
                line.set_data([], [])

        ax.set_title(f"Joueur {index_player} - frame {frame_idx}")
        return (scat, *lines)

    anim = FuncAnimation(
        fig,
        update,
        frames=T,
        init_func=init,
        interval=40,
        blit=True,
        repeat=True,

    )

    plt.show()
    return anim



def visualize_3D_npy(npy_path: Union[str, Path], index_player: int):
    """
    Charge un fichier .npy contenant des poses 3D de forme (T, N, 25, 3)
    et affiche une animation matplotlib 3D des 25 keypoints d'un joueur donné,
    avec segments entre keypoints.

    Paramètres
    ----------
    npy_path : str | Path
        Chemin vers le fichier .npy.
    index_player : int
        Index du joueur à visualiser (dimension N).

    Retour
    ------
    anim : matplotlib.animation.FuncAnimation
        Objet animation matplotlib.
    """
    npy_path = Path(npy_path)
    data = np.load(npy_path)

    if data.ndim != 4:
        raise ValueError(
            f"Le tableau doit être de dimension 4 (T, N, 25, 3), reçu: {data.shape}"
        )

    T, N, K, D = data.shape

    if K != 25 or D != 3:
        raise ValueError(f"Shape attendue (_, _, 25, 3), reçu: {data.shape}")

    if not (0 <= index_player < N):
        raise IndexError(
            f"index_player={index_player} hors limites. "
            f"Nombre de joueurs disponible dans le tenseur: N={N}"
        )

    player_data = data[:, index_player, :, :]  # shape: (T, 25, 3)


    # Bornes globales sur toutes les frames valides du joueur
    x_all = player_data[..., 0]
    y_all = player_data[..., 1]
    z_all = player_data[..., 2]

    valid_x = x_all[~np.isnan(x_all)]
    valid_y = y_all[~np.isnan(y_all)]
    valid_z = z_all[~np.isnan(z_all)]

    if valid_x.size == 0 or valid_y.size == 0 or valid_z.size == 0:
        raise ValueError(
            f"Le joueur {index_player} est absent de toutes les frames (que des NaN)."
        )

    x_min, x_max = valid_x.min(), valid_x.max()
    y_min, y_max = valid_y.min(), valid_y.max()
    z_min, z_max = valid_z.min(), valid_z.max()

    margin_x = max((x_max - x_min) * 0.05, 1.0)
    margin_y = max((y_max - y_min) * 0.05, 1.0)
    margin_z = max((z_max - z_min) * 0.05, 1.0)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # scatter 3D initial vide
    scat = ax.scatter([], [], [], s=25)

    # Une ligne par segment
    lines = [ax.plot([], [], [], linewidth=2)[0] for _ in SKELETON]

    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)
    ax.set_zlim(z_min - margin_z, z_max + margin_z)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Joueur {index_player} - frame 0")

    # Pour éviter les déformations visuelles trop fortes
    try:
        ax.set_box_aspect((
            (x_max - x_min) + 2 * margin_x,
            (y_max - y_min) + 2 * margin_y,
            (z_max - z_min) + 2 * margin_z,
        ))
    except AttributeError:
        pass

    def init():
        scat._offsets3d = ([], [], [])
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return (scat, *lines)

    def update(frame_idx: int):
        points = player_data[frame_idx]  # (25, 3)

        valid_mask = ~np.isnan(points).any(axis=1)
        valid_points = points[valid_mask]

        if valid_points.size == 0:
            scat._offsets3d = ([], [], [])
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            ax.set_title(f"Joueur {index_player} - frame {frame_idx} (absent)")
            return (scat, *lines)

        scat._offsets3d = (
            valid_points[:, 0],
            valid_points[:, 1],
            valid_points[:, 2],
        )

        for line, (i, j) in zip(lines, SKELETON):
            if valid_mask[i] and valid_mask[j]:
                xs = [points[i, 0], points[j, 0]]
                ys = [points[i, 1], points[j, 1]]
                zs = [points[i, 2], points[j, 2]]
                line.set_data(xs, ys)
                line.set_3d_properties(zs)
            else:
                line.set_data([], [])
                line.set_3d_properties([])

        ax.set_title(f"Joueur {index_player} - frame {frame_idx}")
        return (scat, *lines)

    anim = FuncAnimation(
        fig,
        update,
        frames=T,
        init_func=init,
        interval=40,
        blit=False,   # en 3D, blit=False est plus robuste
        repeat=True,
    )

    plt.show()
    return anim

#anim = visualize_2D_npy("data/skel_2d/ARG_CRO_000737.npy", index_player=1)
anim = visualize_3D_npy("data/skel_3d/ARG_CRO_000737.npy", index_player=0)