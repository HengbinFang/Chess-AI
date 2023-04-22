
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from CCRLDataset import CCRLDataset
from AlphaZeroNetwork import AlphaZeroNet

#Training params
num_epochs = 40
num_blocks = 10
num_filters = 128
ccrl_dir = 'reformat'
logmode = True
cuda = True

"""
Training loop:


"""
def train():
    train_ds = CCRLDataset(ccrl_dir) # Load dataset

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=48) # Put dataset in a DataLoader
    alphaZeroNet = AlphaZeroNet(num_blocks, num_filters) # Create Alpha Zero Net
    if cuda:
        alphaZeroNet = alphaZeroNet.cuda() # Switch to CUDA
        
    optimizer = optim.Adam( alphaZeroNet.parameters() ) 
    mseLoss = nn.MSELoss()

    print('Starting training')
    for epoch in range(num_epochs):
        alphaZeroNet.train() # Train mode
        for iter_num, data in enumerate(train_loader):
            optimizer.zero_grad() # Reset Grad

            position = data['position'] # The encoded position of the game the AI can understand
            valueTarget = data['value'] # Winner of game
            policyTarget = data['policy'] # Index of the encoded target move

            if cuda: # Transfer data on CUDA for fast calculations
                position = position.cuda()
                valueTarget = valueTarget.cuda()
                policyTarget = policyTarget.cuda()

            valueLoss, policyLoss = alphaZeroNet(
                position,
                valueTarget=valueTarget,
                policyTarget=policyTarget
            )

            loss = valueLoss + policyLoss
            # valueLoss: MSELoss
            # policyLoss: CrossEntropyLoss

            loss.backward() # WTF???!?!?!
            optimizer.step() # Adam step

            message = 'Epoch {:03} | Step {:05} / {:05} | Value loss {:0.5f} | Policy loss {:0.5f}'.format(
                     epoch, iter_num, len( train_loader ), float( valueLoss ), float( policyLoss ) )
            
            if iter_num != 0 and not logmode:
                print( ('\b' * len(message) ), end='' )
            print( message, end='', flush=True )
            if logmode:
                print('')
        
        print('') # Save Model
        networkFileName = 'AlphaZeroNet_{}x{}.pt'.format( num_blocks, num_filters ) 
        torch.save( alphaZeroNet.state_dict(), networkFileName )
        print( 'Saved model to {}'.format( networkFileName ) )

if __name__ == '__main__':
    train()
