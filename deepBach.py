"""
@author: Gaetan Hadjeres
"""

import click
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata

from DeepBach.model_manager import DeepBach
import torch

@click.command()
@click.option('--note_embedding_dim', default=64,
              help='size of the note embeddings')
@click.option('--meta_embedding_dim', default=64,
              help='size of the metadata embeddings')
@click.option('--num_layers', default=2,
              help='number of layers of the LSTMs')
@click.option('--lstm_hidden_size', default=256,
              help='hidden size of the LSTMs')
@click.option('--dropout_lstm', default=0.5,
              help='amount of dropout between LSTM layers')
@click.option('--linear_hidden_size', default=256,
              help='hidden size of the Linear layers')
@click.option('--batch_size', default=256,
              help='training batch size')
@click.option('--num_epochs', default=5,
              help='number of training epochs')
@click.option('--train', is_flag=True,
              help='train the specified model for num_epochs')
@click.option('--num_iterations', default=500,
              help='number of parallel pseudo-Gibbs sampling iterations')
@click.option('--sequence_length_ticks', default=512,
              help='length of the generated chorale (in ticks)')
def main(note_embedding_dim,
         meta_embedding_dim,
         num_layers,
         lstm_hidden_size,
         dropout_lstm,
         linear_hidden_size,
         batch_size,
         num_epochs,
         train,
         num_iterations,
         sequence_length_ticks,
         ):
    dataset_manager = DatasetManager()

    metadatas = [
       FermataMetadata(),
       TickMetadata(subdivision=4),
       KeyMetadata()
    ]
    chorale_dataset_kwargs = {
        'voice_ids':      [1,2],
        'metadatas':      metadatas,
        'sequences_size': 8,
        'subdivision':    4
    }
    bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(
        name='bach_chorales',
        **chorale_dataset_kwargs
        )
    print("not downloaded yet")
    dataset = bach_chorales_dataset
    print("dataset",type(bach_chorales_dataset),bach_chorales_dataset.corpus_it_gen)
    print(dataset.note2index_dicts[0],dataset.note2index_dicts[0])
    # print(dataset.tensor_dataset)
    # exit()
    deepbach = DeepBach(
        dataset=dataset,
        note_embedding_dim=note_embedding_dim,
        meta_embedding_dim=meta_embedding_dim,
        num_layers=num_layers,
        lstm_hidden_size=lstm_hidden_size,
        dropout_lstm=dropout_lstm,
        linear_hidden_size=linear_hidden_size
    )

    # print(type(dataset))
    # exit

    if train:
        deepbach.train(batch_size=batch_size,
                       num_epochs=num_epochs)
    else:
        deepbach.load()
        deepbach.cuda()



    print('Generation')
    t = torch.tensor([[ 4,  2,  2,  2, -1,  -1,  -1,  -1, -1,  -1,  -1,  2,  -1,  -1, -1,  2,  5,  2,
          2,  2, 29,  -1,  -1,  -1,  3,  2, -1,  -1,  0,  2,  2,  -1,  4,  2,  2,  2,
         16,  2,  2,  -1,  -1,  -1,  52,  2,  -1,  2,  2,  21, 20,  -1,  2,  2, 26,  2,
          2,  2, 33,  -1,  2,  -1, 16,  -1,  -1,  -1],
        [24,  -1,  -1,  -1, 31,  -1,  2,  2, -1,  2,  2,  2,  5,  -1,  2,  2,  5,  2,
          2,  -1,  5,  -1,  7,  -1,  5,  2, -1,  2,  3,  2, 31,  -1,  5,  2,  2,  2,
         11,  -1, 28,  -1,  5,  -1,  2,  2,  -1,  2,  2,  2, 28,  -1,  2,  2,  3,  2,
          2,  -1, 35,  -1,  2,  -1, 31,  2,  -1,  -1],])
        # [24,  -1,  -1,  -1, 11,  -1,  -1,  1, 11,  1, -1,  -1, -1,  -1,  -1,  -1, 23,  1,
        #   1,  -1, 17,  -1, 14,  -1, -1,  18,  1,  1,  -1,  -1,  -1,  -1, -1,  -1, 23, 27,
        #  23,  1,  1,  -1, -1,  -1,  -1,  1,  1,  1,  -1,  -1, -1,  -1,  -1,  -1, 17,  1,
        #   1,  1, 35,  -1,  1,  -1, -1,  -1,  -1,  -1],
        # [ 15,  -1, 39,  -1, 33,  -1, -1,  2,  1,  2,  -1,  2, -1,  2, -1,  -1, -1,  2,
        #   -1,  -1, 13,  -1,  1,  -1, -1,  2,  2,  2,  -1,  2, -1,  2, -1,  -1,  -1,  2,
        #   -1,  -1,  2,  -1, 12,  -1,  -1,  2,  2,  2,  -1,  2, -1,  2,  -1,  -1, -1,  2,
        #   -1,  -1, 45,  -1, 22,  -1,  -1,  -1,  -1,  -1]])

    score, tensor_chorale, tensor_metadata = deepbach.generation(
        num_iterations=num_iterations,
        sequence_length_ticks=sequence_length_ticks,
        # tensor_chorale = t

    )
    print(tensor_chorale)
    print(tensor_metadata)
    # exit()
    score.show('txt')
    score.show()


if __name__ == '__main__':
    main()
