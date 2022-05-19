from gfpgan import GFPGANer
import towhee

@towhee.register
class GFPGANerOp:

    def __init__(self,
                 model_path='/GFPGAN.pth',
                 upscale=2,
                 arch='clean',
                 channel_multiplier=2,
                 bg_upsampler=None) -> None:
        self._restorer = GFPGANer(model_path, upscale, arch, channel_multiplier, bg_upsampler)

    def __call__(self, img):
        cropped_faces, restored_faces, restored_img = self._restorer.enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True)

        return restored_faces[0][:, :, ::-1]

(
    towhee.glob['path']('*.jpg') 
        .image_load['path', 'img']() 
        .GFPGANerOp['img','face']() 
        .show(formatter=dict(img='image', face='image'))
)