import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import argparse
from tqdm import tqdm
import json

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import Encoder, Decoder
from transformer_exploration import AlibiEncoder, AlibiDecoder
from utilities import Utilities

seed = 42
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training


data_path = 'speechesdataset'
cls_train_data_path = "speechesdataset/train_CLS.tsv"
cls_validation_data_path = "speechesdataset/test_CLS.tsv"
lm_train_data_path = "speechesdataset/train_LM.txt"
lm_obama_data_path = "speechesdataset/test_LM_obama.txt"
lm_hbush_data_path = "speechesdataset/test_LM_hbush.txt"
lm_wbush_data_path = "speechesdataset/test_LM_wbush.txt"


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename or ".ipynb_checkpoints" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

# Initialize all tokenizers, datasets and dataloaders
print("Loading data and creating tokenizer ...")
texts = load_texts(data_path)
tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
print("Vocabulary size is", tokenizer.vocab_size)
print()

train_CLS_dataset = SpeechesClassificationDataset(tokenizer, cls_train_data_path)
train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch)

val_CLS_dataset = SpeechesClassificationDataset(tokenizer, cls_validation_data_path)
val_CLS_loader = DataLoader(val_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch)

train_LM_dataset = LanguageModelingDataset(tokenizer, lm_train_data_path, block_size)
train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

test_LM_Obama_dataset = LanguageModelingDataset(tokenizer, lm_obama_data_path, block_size)
test_LM_Obama_loader = DataLoader(test_LM_Obama_dataset, batch_size=batch_size, shuffle=True)

test_LM_Hbush_dataset = LanguageModelingDataset(tokenizer, lm_hbush_data_path, block_size)
test_LM_Hbush_loader = DataLoader(test_LM_Hbush_dataset, batch_size=batch_size, shuffle=True)

test_LM_Wbush_dataset = LanguageModelingDataset(tokenizer, lm_wbush_data_path, block_size)
test_LM_Wbush_loader = DataLoader(test_LM_Wbush_dataset, batch_size=batch_size,collate_fn=collate_batch)


def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _ = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss, _ = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        # total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def get_total_params(model):
    """Returns the total number of parameters in a PyTorch model."""
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params

def part1():

    # for the classification  task, you will train for a fixed number of epochs like this:
    cls_model = Encoder(vocab_size= tokenizer.vocab_size, 
                    embedding_size=n_embd, 
                    num_layers=n_layer, 
                    block_size=block_size, 
                    num_heads=n_head, 
                    dropout=0.2,
                    classifier_hidden_dim = n_hidden,
                    classifier_dim=n_output)
    cls_model = cls_model.to(device)
    cls_optimizer = torch.optim.Adam(cls_model.parameters(), lr=learning_rate)

    print("Training Classifier")
    for epoch in range(epochs_CLS):
        epoch_loss = 0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            # CLS training code here
            prediction, _ = cls_model(xb)
            loss = torch.nn.functional.cross_entropy(prediction, yb)
            cls_optimizer.zero_grad()
            loss.backward()
            cls_optimizer.step()

            epoch_loss += loss.item()

        train_acc = compute_classifier_accuracy(cls_model, train_CLS_loader)
        val_acc = compute_classifier_accuracy(cls_model, val_CLS_loader)
        print(f"Epoch: {epoch} | Train Acc: {train_acc} | Train Loss: {epoch_loss/len(train_CLS_loader)} | Val Acc: {val_acc}")

    val_acc = compute_classifier_accuracy(cls_model, val_CLS_loader)
    print(f"Final Classification Results | Val Acc: {val_acc}")
    print()

    print("Performing Sanity Check on Encoder...")
    encoder_sanity_checker = Utilities(tokenizer, cls_model)
    encoder_sanity_checker.sanity_check("Today, old adversaries are at peace, and emerging democracies are potential partners.", block_size, "e1")
    encoder_sanity_checker.sanity_check("We're going to spend more on our schools, and we're going to spend it more wisely.", block_size, "e2")
    print("Encoder Sanity Check Complete!")
    print()

def part2():

    lm_model = Decoder(vocab_size=tokenizer.vocab_size,
                    embedding_size=n_embd,
                    num_layers=n_layer, 
                    block_size=block_size, 
                    num_heads=n_head, 
                    dropout=0.2,
                    )
    lm_model = lm_model.to(device)
    lm_optimizer = torch.optim.AdamW(lm_model.parameters(), lr=learning_rate)

    print("Training Language Model")
    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i > max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        # LM training code here
        loss, _ = lm_model(xb, yb)
        lm_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        lm_optimizer.step()

        if i%eval_interval == 0:
            train_perplexity = compute_perplexity(lm_model, train_LM_loader, eval_iters)
            print(f"Iter: {i} | Train Perplexity: {train_perplexity}") 


    test_obama_perplexity = compute_perplexity(lm_model, test_LM_Obama_loader, eval_iters)
    test_hbush_perplexity = compute_perplexity(lm_model, test_LM_Hbush_loader, eval_iters)
    test_wbush_perplexity = compute_perplexity(lm_model, test_LM_Wbush_loader, eval_iters)
    print(f"Final LM Results (Perplexity) | Train: {train_perplexity} | Test Obama: {test_obama_perplexity} | Test HBush: {test_hbush_perplexity} | Test WBush: {test_wbush_perplexity}") 
    print()

    print("Performing Sanity Check on Decoder...")
    decoder_sanity_checker = Utilities(tokenizer, lm_model)
    decoder_sanity_checker.sanity_check("Today, old adversaries are at peace, and emerging democracies are potential partners.", block_size, "d1")
    decoder_sanity_checker.sanity_check("We're going to spend more on our schools, and we're going to spend it more wisely.", block_size, "d2")
    print("Decoder Sanity Check Complete!")


def part3a():
    
    # for the classification  task, you will train for a fixed number of epochs like this:
    cls_model = AlibiEncoder(vocab_size= tokenizer.vocab_size, 
                    embedding_size=n_embd, 
                    num_layers=n_layer, 
                    block_size=block_size, 
                    num_heads=n_head, 
                    dropout=0.2,
                    classifier_hidden_dim = n_hidden,
                    classifier_dim=n_output)
    cls_model = cls_model.to(device)
    cls_optimizer = torch.optim.Adam(cls_model.parameters(), lr=learning_rate)

    print("Training Classifier: Alibi Implementation")
    for epoch in range(epochs_CLS):
        epoch_loss = 0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            # CLS training code here
            prediction, _ = cls_model(xb)
            loss = torch.nn.functional.cross_entropy(prediction, yb)
            cls_optimizer.zero_grad()
            loss.backward()
            cls_optimizer.step()

            epoch_loss += loss.item()

        train_acc = compute_classifier_accuracy(cls_model, train_CLS_loader)
        val_acc = compute_classifier_accuracy(cls_model, val_CLS_loader)
        print(f"Epoch: {epoch} | Train Acc: {train_acc} | Train Loss: {epoch_loss/len(train_CLS_loader)} | Val Acc: {val_acc}")

    val_acc = compute_classifier_accuracy(cls_model, val_CLS_loader)
    print(f"Final Classification Results | Val Acc: {val_acc}")
    print()

    print("Performing Sanity Check on Alibi Encoder...")
    encoder_sanity_checker = Utilities(tokenizer, cls_model)
    encoder_sanity_checker.sanity_check("Today, old adversaries are at peace, and emerging democracies are potential partners.", block_size, "e1_alibi")
    encoder_sanity_checker.sanity_check("We're going to spend more on our schools, and we're going to spend it more wisely.", block_size, "e2_alibi")
    print("Alibi Encoder Sanity Check Complete!")
    print()

    lm_model = AlibiDecoder(vocab_size=tokenizer.vocab_size,
                    embedding_size=n_embd,
                    num_layers=n_layer, 
                    block_size=block_size, 
                    num_heads=n_head, 
                    dropout=0.2,
                    )
    lm_model = lm_model.to(device)
    lm_optimizer = torch.optim.AdamW(lm_model.parameters(), lr=learning_rate)

    print("Training Language Model: Alibi Implementation")
    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i > max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        # LM training code here
        loss, _ = lm_model(xb, yb)
        lm_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        lm_optimizer.step()

        if i%eval_interval == 0:
            train_perplexity = compute_perplexity(lm_model, train_LM_loader, eval_iters)
            print(f"Iter: {i} | Train Perplexity: {train_perplexity}") 


    test_obama_perplexity = compute_perplexity(lm_model, test_LM_Obama_loader, eval_iters)
    test_hbush_perplexity = compute_perplexity(lm_model, test_LM_Hbush_loader, eval_iters)
    test_wbush_perplexity = compute_perplexity(lm_model, test_LM_Wbush_loader, eval_iters)
    print(f"Final LM Results (Perplexity) | Train: {train_perplexity} | Test Obama: {test_obama_perplexity} | Test HBush: {test_hbush_perplexity} | Test WBush: {test_wbush_perplexity}") 
    print()

    print("Performing Sanity Check on Alibi Decoder...")
    decoder_sanity_checker = Utilities(tokenizer, lm_model)
    decoder_sanity_checker.sanity_check("Today, old adversaries are at peace, and emerging democracies are potential partners.", block_size, "d1_alibi")
    decoder_sanity_checker.sanity_check("We're going to spend more on our schools, and we're going to spend it more wisely.", block_size, "d2_alibi")
    print("Alibi Decoder Sanity Check Complete!")
    
    
def train_cls(cls_model, cls_optimizer, tokenizer):
    
    all_train_acc = []
    all_val_acc = []
    
    print("Training Classifier")
    for epoch in tqdm(range(epochs_CLS)):
        epoch_loss = 0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            # CLS training code here
            prediction, _ = cls_model(xb)
            loss = torch.nn.functional.cross_entropy(prediction, yb)
            cls_optimizer.zero_grad()
            loss.backward()
            cls_optimizer.step()
            
            epoch_loss += loss.item()
            
        train_acc = compute_classifier_accuracy(cls_model, train_CLS_loader)
        val_acc = compute_classifier_accuracy(cls_model, val_CLS_loader)
        all_train_acc.append(train_acc)
        all_val_acc.append(val_acc)
    
    val_acc = compute_classifier_accuracy(cls_model, val_CLS_loader)
    print(f"Final Classification Results | Val Acc: {val_acc}")
    print()
    return all_train_acc, all_val_acc

def part3_b(explore=True, lr=1e-3):
    # Model Performance Improvement
    
    if explore:
        # exploring effect of depth of the network
        model_depth = [3, 4, 5]
        depth_results = {}
        for layers in model_depth:
            print("Model Depth", layers)
            cls_model = Encoder(vocab_size= tokenizer.vocab_size, 
                                embedding_size=n_embd, 
                                num_layers=layers, # changed
                                block_size=block_size, 
                                num_heads=n_head, 
                                dropout=0.2,
                                classifier_hidden_dim = n_hidden,
                                classifier_dim=n_output)
            cls_model = cls_model.to(device)

            # Different Optimizer
            cls_optimizer = torch.optim.AdamW(cls_model.parameters(), lr=lr)

            train, val = train_cls(cls_model, cls_optimizer, tokenizer)
            depth_results[layers] = [train, val]

        with open(f"Results/{lr}_model_depth_analysis.json", "w") as f:
            json.dump(depth_results, f, indent=4)

        # exploring effect of classifier hidden dim
        cls_hidden_dim = [64, 128, 256]
        cls_hidden_dim_results = {}
        for dim in cls_hidden_dim:
            print("Hidden Dimension:", dim)
            cls_model = Encoder(vocab_size= tokenizer.vocab_size, 
                                embedding_size=n_embd, 
                                num_layers=n_layer, 
                                block_size=block_size, 
                                num_heads=n_head, 
                                dropout=0.2,
                                classifier_hidden_dim = dim, # changed
                                classifier_dim=n_output)
            cls_model = cls_model.to(device)

            # Different Optimizer
            cls_optimizer = torch.optim.AdamW(cls_model.parameters(), lr=lr)

            train, val = train_cls(cls_model, cls_optimizer, tokenizer)
            cls_hidden_dim_results[dim] = [train, val]

        with open(f"Results/{lr}_cls_hidden_dim_analysis.json", "w") as f:
            json.dump(cls_hidden_dim_results, f, indent=4)

        # exploring effect of multihead attention heads
        heads = [2, 4, 8]
        head_results = {}
        for head in heads:
            print("Num Heads", head)
            cls_model = Encoder(vocab_size= tokenizer.vocab_size, 
                                embedding_size=n_embd, 
                                num_layers=n_layer, 
                                block_size=block_size, 
                                num_heads=head, # changed
                                dropout=0.2,
                                classifier_hidden_dim = n_hidden,
                                classifier_dim=n_output)
            cls_model = cls_model.to(device)

            # Different Optimizer
            cls_optimizer = torch.optim.AdamW(cls_model.parameters(), lr=lr)

            train, val = train_cls(cls_model, cls_optimizer, tokenizer)
            head_results[head] = [train, val]

        with open(f"Results/{lr}_head_analysis.json", "w") as f:
            json.dump(head_results, f, indent=4) 
        
    # best paramter only
    
    if not explore:
        best_results = {}
        cls_model = Encoder(vocab_size= tokenizer.vocab_size, 
                            embedding_size=n_embd, 
                            num_layers=4, 
                            block_size=block_size, 
                            num_heads=4,
                            dropout=0.2,
                            classifier_hidden_dim = 100,
                            classifier_dim=n_output)
        cls_model = cls_model.to(device)
        
        # Different Optimizer
        cls_optimizer = torch.optim.AdamW(cls_model.parameters(), lr=3e-3)
        
        train, val = train_cls(cls_model, cls_optimizer, tokenizer)
        best_results['train'] = train
        best_results['val'] = val
        with open("Results/best_model.json", "w") as f:
            json.dump(best_results, f, indent=4) 


def main(args):    
    if args.task == 'part1':
        part1()
    
    elif args.task == 'part2':
        part2()
        
    elif args.task == 'part3a':
        print("Alibi Implementation | Architecture Exploration")
        part3a()
    
    elif args.task == 'part3b':
        print("Exploring Trends | Learning Rate: 3e-3")
        part3_b(True, 3e-3)
        print("Exploring Trends | Learning Rate: 1e-3")
        part3_b(True, 1e-3)
        
    elif args.task == 'part3b_best':
        print("Best Model")
        part3_b(False, 3e-3)
    
    else:
        print("Invalid option {args.task} selected.")
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program to run different tasks.")
    parser.add_argument("task", type=str, help="Task to run: 'part1', 'part2', 'part3a', 'part3b', or 'part3b_best")
    args = parser.parse_args()
    main(args)
