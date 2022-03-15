import torch
import torchvision
from data import load_cifar10
import argparse
from tqdm import tqdm
import os


#TODO : 
#1-gradient clipping
#2-tensorboard
def train(model, train_loader , val_loader, cfg):
    """
       Simple training loop for PyTorch model.
       cfg: criterion, optimizer ,epochs , model_path='model.ckpt' , scheduler=None  ,load_model=False, min_val_acc_to_save=88.0

    """ 
    if cfg['gpu']:
      device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    best_val_acc=0
    # Make sure model is in training mode.
    if cfg['Load_model'] and cfg['load_path']:
      print('Loading the model from ckpt.....')
      train_ckpt=torch.load(cfg['load_path'])
      model.load_state_dict(train_ckpt['model'])
      print('The model is ready!')


    model.train()
    cfg['optimizer'].zero_grad()

    # Move model to the device (CPU or GPU).
    model.to(device)
    
    # Exponential moving average of the loss.
    ema_loss = None
    losses=[]
    train_accs=[]
    val_accs=[]

    print(f'----- Training on {device} -----')
    # Loop over epochs.
    for epoch in range(cfg['epochs']):
        correct = 0
        num_examples=0
        # Loop over data.
        loop=tqdm(enumerate(train_loader , start =epoch*len(train_loader)), total=len(train_loader))
        for step , (images, target) in loop:
            # Forward pass.
            output = model(images.to(device))
            loss = cfg['criterion'](output.to(device), target.to(device))

            # Backward pass.
            loss = loss / cfg['accumulation_steps'] # Normalize our loss (if averaged)
            loss.backward()
            if epoch+1 % cfg['accumulation_steps']==0:
              cfg['optimizer'].step()
              cfg['optimizer'].zero_grad()


            # NOTE: It is important to call .item() on the loss before summing.
            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss += (loss.item() - ema_loss) * 0.01 
            # Compute the correct classifications
            preds = output.argmax(dim=1, keepdim=True)
            correct+= preds.cpu().eq(target.view_as(preds)).sum().item()
            num_examples+= images.shape[0]
            train_acc=correct/num_examples
            #tqdm
            loop.set_description(f"Epoch [{epoch+1}/{cfg['epochs']}]")
            loop.set_postfix(loss=ema_loss, acc=train_acc)
        
        losses.append(ema_loss)
        train_accs.append(train_acc)
        #schedular
        if cfg['scheduler']:
          cfg['scheduler'].step()
        #validate
        if epoch+1 % cfg['val_period']==0:
          val_acc = test(model ,val_loader, device)
          val_accs.append(val_acc)
          if val_acc > best_val_acc and val_acc > cfg['min_val_acc_to_save']:
              print(f'validation accuracy increased from {best_val_acc} to {val_acc}  , saving the model ....')
              #saving training ckpt
              chk_point={'model_sate_dict':model.state_dict(), 'epochs':epoch+1, 'best_val_acc':best_val_acc}
              torch.save(chk_point, cfg['ckpt_path'])
              best_val_acc=val_acc
        print('-------------------------------------------------------------')

        return train_accs , val_accs, losses


if __name__=="__main__":


    parser = argparse.ArgumentParser(description='Train SimCLR on CIFAR-10')


    parser.add_argument('--ckpt-path', default='model.ckpt')
    parser.add_argument('--load-path', default='model.ckpt')

    parser.add_argument('--lr', '--learning-rate-weights', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epochs', default=5, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--seed', default=44, type=int)
    parser.add_argument('--workers', default=4, type=int)

    parser.add_argument('--accumulation-steps', default=1, type=int)
    parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size')

    parser.add_argument('--Load-model', default=False, type=bool)
    parser.add_argument('--gpu', default=True, type=bool)
    parser.add_argument('--min-val-acc-to-save', default=30.0, type=float)

    parser.add_argument('--val-period', default=1, type=int)

    args = parser.parse_args('')  # running in ipynb

    if __name__ == '__main__':

        model = torchvision.models.resnet50()
        model.fc=torch.nn.Linear(2048,10)
        train_loader , val_loader = load_cifar10()
        args.optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr , momentum=args.momentum)
        args.criterion = torch.nn.CrossEntropyLoss()
        args.scheduler = torch.optim.lr_scheduler.StepLR(args.optimizer, step_size=3, gamma=0.1)
        print('training....')
        train(model, train_loader , val_loader, cfg=vars(args))