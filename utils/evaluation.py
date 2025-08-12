from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from sklearn.metrics.pairwise import cosine_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms

def compute_PSNR(img_true, img_test):
    return peak_signal_noise_ratio(img_true, img_test)

def compute_SSIM(img1, img2):
    data_range = img1.max() - img1.min()

    return structural_similarity(img1, img2, channel_axis=2, data_range=data_range)

def compute_id_cosine_similarity(id_img, out_img, pl_module):
    recognition_model = pl_module.recognition_model
    transform = transforms.Compose([transforms.ToTensor()])

    id_img = transform(id_img.copy())
    out_img = transform(out_img.copy())

    xid_feature, _ = recognition_model(id_img.unsqueeze(0).cuda())
    out_feature, _ = recognition_model(out_img.unsqueeze(0).cuda())
    cs = cosine_similarity(xid_feature.cpu(), out_feature.cpu()).mean()
    return cs

def compute_lpips(exp_img, out_img):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    exp_img = transform(exp_img.copy()).unsqueeze(0)
    out_img = transform(out_img.copy()).unsqueeze(0)

    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    return lpips(exp_img, out_img)