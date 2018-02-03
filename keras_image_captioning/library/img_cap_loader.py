import os


def load_img_cap(img_dir_path, txt_dir_path):
    images = dict()
    texts = dict()
    for f in os.listdir(img_dir_path):
        if os.path.isfile(f) and f.endswith('.png'):
            name = f.replace('.png', '')
            images[name] = os.path.join(img_dir_path, f)
    for f in os.listdir(txt_dir_path):
        if os.path.isfile(f) and f.endswith('.txt'):
            name = f.replace('.txt', '')
            texts[name] = open(os.path.join(txt_dir_path, f), 'rt').read()

    result = []
    for name, img_path in images.items():
        if name in texts:
            text = texts[name]
            result.append([img_path, text])

    return result
