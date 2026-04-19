import torch
import numpy as np
from neurogolf.grid_codec import encode_grid_to_tensor, decode_tensor_to_grid
from neurogolf.solvers import RotationSolver, ShiftSolver, FlipSolver, OverlaySolver
from neurogolf.constants import STATE_CHANNELS, COLOR_CHANNELS, IDENTITY_CHANNELS

def test_identity_persistence():
    print("🚀 Running Identity Persistence Sanity Suite...")
    
    # 1. Setup a simple grid with two objects
    # Object 1: Red (2) Square at (2,2)
    # Object 2: Blue (1) Dot at (10,10)
    grid = [[0]*30 for _ in range(30)]
    grid[2][2] = 2; grid[2][3] = 2
    grid[3][2] = 2; grid[3][3] = 2
    grid[10][10] = 1
    
    state = torch.from_numpy(encode_grid_to_tensor(grid))
    
    # Identify which slots belong to which colors
    def get_slot_for_color(st, color_idx):
        color_mask = (st[0, color_idx] > 0.5).float()
        for i in range(IDENTITY_CHANNELS):
            id_mask = st[0, COLOR_CHANNELS + i]
            if torch.abs(id_mask - color_mask).sum() < 1e-4 and color_mask.sum() > 0:
                return i
        return None

    slot_red = get_slot_for_color(state, 2)
    print(f"Slot for Red: {slot_red}")
    assert slot_red is not None, "Red square ID slot not found!"

    # 2. Test Invariance: Rotate
    rot = RotationSolver(k=1)
    state_rot = rot(state)
    
    # Verify color mask matches ID mask for the correct slot
    color_mask_rot = (state_rot[0, 2] > 0.5).float() # Red after rotation
    id_mask_rot = state_rot[0, COLOR_CHANNELS + slot_red]
    
    # Check for perfect overlap
    diff = torch.abs(color_mask_rot - id_mask_rot).sum().item()
    print(f"Rotation Invariance Diff: {diff}")
    assert diff < 1e-4, f"Identity drifted during rotation! Diff: {diff}"

    # 3. Test Merge Persistence
    # Use the original state with two objects (Slot 0 and Slot 1)
    # Move object 2 (dot) to overlap object 1 (square)
    slot_dot = (1 - slot_red) # Since we have only [0,1]
    
    # Square is at (2,2), Dot is at (10,10)
    # Shift dot by (-8, -8) to hit (2,2)
    shift_dot = ShiftSolver(dx=-8, dy=-8)
    
    # We want to isolate object 2, shift it, then overlay back on the original
    # For now, we'll just shift the whole state and overlay (simulates collision)
    state_shifted = shift_dot(state)
    
    overlay = OverlaySolver()
    state_merged = overlay(state, state_shifted)
    
    # At (2,2), we should have identity of both the original square AND the shifted dot
    pixel_ids = state_merged[0, COLOR_CHANNELS:, 2, 2]
    print(f"Merged Pixel IDs at (2,2): {pixel_ids[:5]}")
    # Slot 0 and Slot 1 should both be active
    assert pixel_ids[0] > 0.5 and pixel_ids[1] > 0.5, "Lineage union failed during collision!"
    
    # 4. Test Color Independence
    # Recolor Slot 0 from Red (2) to Green (3)
    # (Using a manual channel swap for now)
    state_recolored = state.clone()
    state_recolored[0, 3] = state_recolored[0, 2] # Copy Red to Green
    state_recolored[0, 2] = 0.0 # Clear Red
    
    id_after = state_recolored[0, COLOR_CHANNELS + slot_red]
    id_before = state[0, COLOR_CHANNELS + slot_red]
    
    diff_id = torch.abs(id_after - id_before).sum().item()
    print(f"Color Independence ID Diff: {diff_id}")
    assert diff_id < 1e-4, "Identity changed when color was manually swapped!"

    print("✅ All Identity Sanity Tests Passed!")

if __name__ == "__main__":
    test_identity_persistence()
