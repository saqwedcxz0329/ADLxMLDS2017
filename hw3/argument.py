def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''

    parser.add_argument('--models_dir', default='models', help='where are the models')
    parser.add_argument('--trained_pg_model_name', default=None, help='pg pretrained model')
    parser.add_argument('--store_pg_model_name', default='model_pg', help='trained pg model to store')

    return parser
