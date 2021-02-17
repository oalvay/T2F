from train_network import *
from pro_gan_pytorch.PRO_GAN import ConditionalProGAN
from networks.TextEncoder import Encoder
from networks.ConditionAugmentation import ConditionAugmentor

th.manual_seed(6666)

config = get_config('configs/3.conf')

c_pro_gan = ConditionalProGAN(
    embedding_size=config.hidden_size,
    depth=config.depth,
    latent_size=config.latent_size,
    compressed_latent_size=config.compressed_latent_size,
    learning_rate=config.learning_rate,
    beta_1=config.beta_1,
    beta_2=config.beta_2,
    eps=config.eps,
    drift=config.drift,
    n_critic=config.n_critic,
    use_eql=config.use_eql,
    loss=config.loss_function,
    use_ema=config.use_ema,
    ema_decay=config.ema_decay,
    device=device
)
c_pro_gan.gen.load_state_dict(th.load('training_runs/3/saved_models/GAN_GEN_4.pth'))
c_pro_gan.dis.load_state_dict(th.load('training_runs/3/saved_models/GAN_DIS_4.pth'))

ca = ConditionAugmentor(
    input_size=config.hidden_size,
    latent_size=config.ca_out_size,
    use_eql=config.use_eql,
    device=device
)
ca.load_state_dict(th.load('training_runs/3/saved_models/Condition_Augmentor_4.pth'))

depth, alpha = 4, 1

dataset = dl.Face2TextDataset(
    pro_pick_file=config.processed_text_file,
    img_dir=config.images_dir,
    img_transform=dl.get_transform(config.img_dims),
    captions_len=config.captions_length
)

encoder = Encoder(
    embedding_size=config.embedding_size,
    vocab_size=dataset.vocab_size,
    hidden_size=config.hidden_size,
    num_layers=config.num_layers,
    device=device
)
encoder.load_state_dict(th.load('training_runs/3/saved_models/Encoder_4.pth'))

temp_data = dl.get_data_loader(dataset, 1, num_workers=3)
fixed_captions, fixed_real_images = iter(temp_data).next()
fixed_embeddings = encoder(fixed_captions)
fixed_embeddings = fixed_embeddings.to(device)
fixed_c_not_hats, _, _ = ca(fixed_embeddings)

fixed_noise = th.randn(len(fixed_captions),
                       c_pro_gan.latent_size - fixed_c_not_hats.shape[-1]).to(device)

fixed_gan_input = th.cat((fixed_c_not_hats, fixed_noise), dim=-1)

samples=c_pro_gan.gen(
    fixed_gan_input, depth, alpha
)

samples_dir = os.path.join(config.sample_dir, "samples")

os.makedirs(samples_dir, exist_ok=True)
create_grid(fixed_real_images, None,  # scale factor is not required here
            os.path.join(samples_dir, "real_samples.png"), real_imgs=True)
create_descriptions_file(os.path.join(samples_dir, "real_captions.txt"),
                         fixed_captions,
                         dataset)


create_grid(
    samples=samples,
    scale_factor=int(np.power(2, c_pro_gan.depth - depth - 1)),
    img_file=os.path.join(samples_dir, "generated_samples.png"),
)

# create_grid(
#     samples=c_pro_gan.gen(
#         fixed_gan_input,
#         current_depth,
#         alpha
#     ),
#     scale_factor=int(np.power(2, c_pro_gan.depth - current_depth - 1)),
#     img_file=gen_img_file,
# )
