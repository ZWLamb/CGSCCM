#把演码和尺度特征图可视化
import os

from PIL import Image
import config

def assemble(seg_dir,scale_dir):
    # 打开两个图像
    image1 = Image.open(seg_dir)
    image2 = Image.open(scale_dir)

    # 确认两个图像的大小是 512x1024
    assert image1.size == (1024, 512)
    assert image2.size == (1024, 512)

    # 创建一个新的空白图像，大小为 1024x1024
    new_image = Image.new('RGB', (1024, 1024))

    # 将第一个图像粘贴到新图像的左侧
    new_image.paste(image1, (0, 0))

    # 将第二个图像粘贴到新图像的右侧
    new_image.paste(image2, (0, 512))

    # 保存合成后的图像
    filename = os.path.basename(seg_dir)
    save_dir = os.path.join('../', config.assemble_dir, 'consep', 'assemble', '{}_assemble.png'.format(filename))
    new_image.save(save_dir)

from pathlib import Path


if __name__ == "__main__":
    seg_dir = os.path.join('../', config.assemble_dir, 'consep', 'results')  # seg_dir
    #scale_dir = os.path.join('../', config.assemble_dir, 'test', 'scale_of_consep')
    path = Path(seg_dir)
    for seg_file in path.rglob('*'):
        if seg_file.is_file():
            if str(seg_file).endswith('.txt'):
                continue
            scale_file = str(seg_file).replace('consep','test').replace('results','scale_of_consep')
            assemble(seg_file, scale_file)



    # filename ="test_11_2.npy"
    # seg_dir = os.path.join('../',config.assemble_dir,'consep','results','{}_sample.png'.format(filename))  #外部调用走这里
    # scale_dir = os.path.join('../',config.assemble_dir,'test','scale_of_consep','{}_sample.png'.format(filename))
    # assemble(seg_dir,scale_dir)


