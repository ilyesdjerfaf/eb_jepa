"""
Visualize DFAUST meshes with proper surface rendering and animation.

Usage:
    uv run python -m examples.mesh_jepa.visualize_dfaust --data_dir datasets/dfaust/raw

    # Render a specific sequence as GIF:
    uv run python -m examples.mesh_jepa.visualize_dfaust --data_dir datasets/dfaust/raw \
        --sequence 50002_jumping_jacks --gif

    # Open interactive 3D viewer:
    uv run python -m examples.mesh_jepa.visualize_dfaust --data_dir datasets/dfaust/raw \
        --sequence 50002_jumping_jacks --interactive
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_sequence(data_dir: Path, sequence_name: str):
    """Load a single sequence and the faces array."""
    faces = None
    vertices = None

    for fname in sorted(data_dir.glob("registrations_*.hdf5")):
        with h5py.File(fname, "r") as f:
            if "faces" in f and faces is None:
                faces = np.array(f["faces"])
            if sequence_name in f:
                vertices = np.array(f[sequence_name]).transpose(2, 0, 1)

    if vertices is None:
        raise ValueError(f"Sequence '{sequence_name}' not found")
    return vertices, faces


def list_sequences(data_dir: Path):
    """List all available sequences."""
    seqs = []
    for fname in sorted(data_dir.glob("registrations_*.hdf5")):
        with h5py.File(fname, "r") as f:
            for key in f.keys():
                if key != "faces":
                    seqs.append((key, f[key].shape[2]))
    return seqs


def render_mesh_frame(ax, vertices, faces, title="", elev=10, azim=135):
    """Render a single mesh frame with triangulated surface.

    DFAUST coordinate system: X=left/right, Y=up/down (height), Z=front/back.
    We plot X→X, Z→Y, Y→Z so the body stands upright in the 3D axes.
    """
    ax.clear()

    # Remap: plot (X, Z, Y) so Y (height) is the vertical axis
    plot_verts = vertices[:, [0, 2, 1]]
    tri_verts = plot_verts[faces]
    mesh = Poly3DCollection(tri_verts, alpha=0.7, linewidths=0.02, edgecolors="gray")

    # Color by height (original Y = plot Z)
    face_colors = plot_verts[faces[:, 0], 2]
    face_colors = (face_colors - face_colors.min()) / (face_colors.max() - face_colors.min() + 1e-8)
    colors = plt.cm.viridis(face_colors)
    mesh.set_facecolor(colors)

    ax.add_collection3d(mesh)

    center = plot_verts.mean(axis=0)
    max_range = (plot_verts.max(axis=0) - plot_verts.min(axis=0)).max() / 2 * 1.1
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def render_static_strip(vertices_seq, faces, n_frames=6, title="", save_path=None):
    """Render N frames side by side with mesh surfaces."""
    T = vertices_seq.shape[0]
    indices = np.linspace(0, T - 1, n_frames, dtype=int)

    fig = plt.figure(figsize=(4 * n_frames, 5))
    fig.suptitle(title, fontsize=14, y=0.95)

    for i, idx in enumerate(indices):
        ax = fig.add_subplot(1, n_frames, i + 1, projection="3d")
        render_mesh_frame(ax, vertices_seq[idx], faces, title=f"frame {idx}")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def render_animated_gif(vertices_seq, faces, save_path, fps=15, stride=2, n_rotations=0):
    """Render sequence as animated GIF."""
    frames_to_render = vertices_seq[::stride]
    n = len(frames_to_render)
    print(f"Rendering {n} frames to GIF (stride={stride})...")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx):
        azim = 135 + (360 * n_rotations * frame_idx / n) if n_rotations else 135
        render_mesh_frame(
            ax, frames_to_render[frame_idx], faces,
            title=f"frame {frame_idx * stride}", azim=azim
        )
        return []

    anim = animation.FuncAnimation(fig, update, frames=n, interval=1000 // fps, blit=False)
    anim.save(str(save_path), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved: {save_path} ({n} frames, {n / fps:.1f}s)")


def render_action_grid(data_dir: Path, faces, subject="50002", save_path=None):
    """Render a grid: rows=actions, columns=time progression."""
    seqs = list_sequences(data_dir)
    subject_seqs = [(name, nf) for name, nf in seqs if name.startswith(subject)]

    n_actions = min(len(subject_seqs), 7)
    n_time = 4

    fig = plt.figure(figsize=(4 * n_time, 3.5 * n_actions))
    fig.suptitle(f"Subject {subject} — action gallery", fontsize=14)

    for row, (seq_name, n_frames) in enumerate(subject_seqs[:n_actions]):
        verts, _ = load_sequence(data_dir, seq_name)
        time_indices = np.linspace(0, len(verts) - 1, n_time, dtype=int)
        action = seq_name.split("_", 1)[1]

        for col, t in enumerate(time_indices):
            ax = fig.add_subplot(n_actions, n_time, row * n_time + col + 1, projection="3d")
            title = f"{action}" if col == 0 else f"t={t}"
            render_mesh_frame(ax, verts[t], faces, title=title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def interactive_viewer(vertices_seq, faces, stride=4):
    """Open an interactive trimesh viewer cycling through frames."""
    import trimesh

    frame_idx = len(vertices_seq) // 2
    mesh = trimesh.Trimesh(vertices=vertices_seq[frame_idx], faces=faces, process=False)
    mesh.visual.vertex_colors = trimesh.visual.interpolate(
        vertices_seq[frame_idx][:, 1], color_map="viridis"
    )
    print(f"Opening viewer at frame {frame_idx}. Close window to exit.")
    mesh.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize DFAUST meshes")
    parser.add_argument("--data_dir", type=str, default="datasets/dfaust/raw")
    parser.add_argument("--save_dir", type=str, default="datasets/dfaust/figures")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Sequence name (e.g. 50002_jumping_jacks). Default: first available")
    parser.add_argument("--gif", action="store_true", help="Render animated GIF")
    parser.add_argument("--interactive", action="store_true", help="Open interactive 3D viewer")
    parser.add_argument("--grid", action="store_true", help="Render action grid for one subject")
    parser.add_argument("--subject", type=str, default="50002", help="Subject for grid view")
    parser.add_argument("--list", action="store_true", help="List all sequences and exit")
    parser.add_argument("--stride", type=int, default=2, help="Frame stride for GIF")
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        seqs = list_sequences(data_dir)
        print(f"{'Sequence':<35} {'Frames':>6}")
        print("-" * 45)
        for name, nf in seqs:
            print(f"{name:<35} {nf:>6}")
        print(f"\nTotal: {len(seqs)} sequences, {sum(nf for _, nf in seqs)} frames")
        return

    # Pick sequence
    if args.sequence is None:
        seqs = list_sequences(data_dir)
        args.sequence = seqs[0][0]
        print(f"No sequence specified, using: {args.sequence}")

    print(f"Loading {args.sequence}...")
    vertices, faces = load_sequence(data_dir, args.sequence)
    print(f"  Shape: {vertices.shape} (frames, vertices, 3)")
    print(f"  Faces: {faces.shape}")

    if args.interactive:
        interactive_viewer(vertices, faces)
        return

    if args.gif:
        gif_path = save_dir / f"{args.sequence}.gif"
        render_animated_gif(vertices, faces, gif_path, stride=args.stride)
        return

    if args.grid:
        grid_path = save_dir / f"action_grid_{args.subject}.png"
        render_action_grid(data_dir, faces, subject=args.subject, save_path=grid_path)
        if not args.no_show:
            plt.show()
        return

    # Default: static strip
    strip_path = save_dir / f"{args.sequence}_strip.png"
    render_static_strip(vertices, faces, n_frames=6,
                        title=args.sequence, save_path=strip_path)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
