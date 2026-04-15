
import torch
model = torch.hub.load(
    "/home/BeeGFS/Laboratories/IBHGC/skhan/Documents/fifa_3DCV_challenge/dinov3",
    "<nom_du_modele_attendu_par_sam3d>",
    source="local",
    pretrained=True,
)
print("DINOv3 OK")