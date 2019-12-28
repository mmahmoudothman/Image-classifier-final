import sys
import argparse
from utils import *
import json

def build_network(arch):
    if arch == 'vgg16':
        net = models.vgg16(pretrained=False)
    elif arch == 'vgg13':
        net = models.vgg13(pretrained=False)
    elif arch == 'vgg11':
        net = models.vgg11(pretrained=False)
    elif arch == 'vgg19':
        net = models.vgg19(pretrained=False)
    else:
        net = models.vgg16(pretrained=False)    
        
    return net

def load_checkpoint(file_name, cuda_enabled):
    map_location = 'cpu'
    if cuda_enabled:
         map_location = 'gpu'
    checkpoint = torch.load(file_name, map_location=map_location)
    num_epochs = checkpoint['num_epochs']
    arch = checkpoint['arch']
    net_rebuild = build_network(arch)
    net_rebuild.load_state_dict(checkpoint['state_dict'])
    class_to_idx = checkpoint['train_datasets_class_to_idx']    
    print('load_checkpoint completed\n')
    return  net_rebuild, class_to_idx, num_epochs

def predict(image_path, cuda_enabled, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.eval()
    image = Image.open(image_path)       
    image = process_image(image)    
    image = image[np.newaxis, :]
    image = image.astype(np.float32)   
    image = torch.from_numpy(image)
    val_input = Variable(image)
    if cuda_enabled:
        val_input.cuda()
    output = model(val_input)
    probability = F.softmax(output.data,dim=1) # use F    
    
    model.train()
    
    return probability

def main(argv): 
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', metavar='input_image', nargs='+', help='image to predict')
    parser.add_argument('checkpoint', metavar='checkpoint', nargs='+', help='full path to load checkpoint')
    parser.add_argument('--top_k', help='how many result to pass back such 3, 5, 10, by default 10', type = int)
    parser.add_argument('--category_names', help='mapping from cat to name')
    parser.add_argument('--gpu', help='enable gpu or not, 1 for enabled, otherwise disabled, by default 0', type = int)
    args = parser.parse_args()  
    
    cuda_enabled = False
    if args.gpu:
        if(args.gpu == 1 and torch.cuda.is_available()):
            cuda_enabled = True
    print("cuda_enabled=\n", cuda_enabled)

    net, class_to_idx, num_epochs = load_checkpoint(args.checkpoint[0], cuda_enabled)
    print("num_epochs=%d" % (num_epochs))
    if cuda_enabled:
        net.cuda()
    # 5. predict
    test_image = args.input_image[0]  
    top_k = 10
    if args.top_k:
        top_k = args.top_k
    print(test_image)
    probability = predict(test_image, cuda_enabled, net, top_k)

    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        probs = np.array(probability.topk(top_k)[0][0])
        index_to_class = {val: key for key, val in class_to_idx.items()} # from reviewer advice
        top_classes = [np.int(index_to_class[each]) for each in np.array(probability.topk(top_k)[1][0])]
        for i in range(top_k):      
            cat = cat_to_name[str(top_classes[i])]
            print('Top [%d], category [%s], probability [%.4f] \n'
                  % (i + 1, cat, probs[i]))
    
    
if __name__ == '__main__':
    main(sys.argv)