import torchvision.transforms.functional as TF


def transform_test(imgs, img_size=256, to_tensor=True, img_size_dynamic=False):
        """
        :param imgs: [ndarray,]
        :return: [ndarray,]
        """
        # resize image and covert to tensor
        #imgs = [TF.to_pil_image(img) for img in imgs]

        if not img_size_dynamic:
            if imgs[0].size != (img_size, img_size):
                imgs = [TF.resize(img, [img_size, img_size], interpolation=3)
                        for img in imgs]

        if to_tensor:
            # to tensor
            imgs = [TF.to_tensor(img) for img in imgs]
            imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    for img in imgs]

        return imgs