import gdown

url = 'https://drive.google.com/uc?id=1-9YjVqgKQsUg2Bi8pg7kuCHecRBcDTKm'
output = 'data/vmax_v0=1.648930169304656.txt'
gdown.download(url, output, quiet=False)