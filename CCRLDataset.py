
import chess.pgn
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import encoder

def tolist( mainline_moves ):
    """
    Change an iterable object of moves to a list of moves.
    
    Args:
        mainline_moves (Mainline object) iterable list of moves

    Returns:
        moves (list of chess.Move) list version of the input moves
    """
    moves = []
    for move in mainline_moves:
        moves.append( move )
    return moves

class CCRLDataset( Dataset ):
    """
    Subclass of torch.utils.data.Dataset for the ccrl dataset.
    """

    def __init__( self, ccrl_dir ):
        """
        Args:
            ccrl_dir (string) Path to directory containing
                pgn files with names 0.pgn, 1.pgn, 2.pgn, etc.
        """
        self.ccrl_dir = ccrl_dir
        self.pgn_file_names = os.listdir( ccrl_dir )

    def __len__( self ):
        """
        Get length of dataset
        """
        return len( self.pgn_file_names )

    def __getitem__( self, idx ):
        """
        Load the game in idx.pgn
        Get a random position, the move made from it, and the winner
        Encode these as numpy arrays
        
        Args:
            idx (int) the index into the dataset.
        
        Returns:
           position (torch.Tensor (16, 8, 8) float32) the encoded position
           policy (torch.Tensor (1) long) the target move's index
           value (torch.Tensor (1) float) the encoded winner of the game
           mask (torch.Tensor (72, 8, 8) int) the legal move mask
        """
        pgn_file_name = self.pgn_file_names[ idx ]
        pgn_file_name = os.path.join( self.ccrl_dir, pgn_file_name ) # Get file name
        with open( pgn_file_name ) as pgn_fh:
            game = chess.pgn.read_game(pgn_fh) # Read PGN file

        moves = tolist(game.mainline_moves()) # Puts all the moves into a list. E.g. [Move.from_uci('e2e4'), ..., Move.from_uci('c4c8')]

        randIdx = int(np.random.random() * ( len( moves ) - 1 )) # Gets a random move's index from the [moves] list.

        board = game.board()

        for idx, move in enumerate(moves): # Moves through each chess piece on the board. UNTIL randIdx has been reached.
            board.push(move)
            if (randIdx == idx): # Checks if the random selected move is the current move
                next_move = moves[idx + 1] # Defines the move AFTER randIdx
                break

        winner = encoder.parseResult(game.headers['Result']) # Gets winner of the SELECTED game

        position, policy, value, mask = encoder.encodeTrainingPoint(board, next_move, winner)
        """
        position = the encoded position that the AI can understand -> encodePosition(board) (?)
        policy = index of the encoded target move
        value = winner of the game
        mask = a mask containing all legal moves (?)
        """
            
        return { 'position': torch.from_numpy( position ),
                 'policy': torch.Tensor( [policy] ).type( dtype=torch.long ),
                 'value': torch.Tensor( [value] ),
                 'mask': torch.from_numpy( mask ) }
