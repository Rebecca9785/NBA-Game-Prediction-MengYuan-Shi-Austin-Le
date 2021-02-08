from src.models.models import *
from src.data.data import *
import sys
import json

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train_model(model, op, epoch, idx_train, adj_hat, features, labels):
    model.train()
    op.zero_grad()
    output = model(features, adj_hat)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    op.step()
    if (epoch + 1) % 50 == 0:
        print('Epoch: {}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()))
    
def test_GraphSage(model, idx_test, labels):
    test_output = model.forward(idx_test).data.numpy().argmax(axis=1)
    loss_test = F.cross_entropy(Variable(torch.from_numpy(test_output)).view(y.size()[0], 1).float(), 
                                labels[idx_test].view(y.size()[0], 1).float())
    acc_test = accuracy(test_output, labels[idx_test])
    
    print("\nTest set results:",
          "loss_test: {:.4f}".format(loss_test.item()),
          "accuracy_test: {:.4f}".format(acc_test.item()))

def test_model(model, idx_test, adj_hat, features, labels):
    model.eval()
    output = model(features, adj_hat)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("\nTest set results:",
          "loss_test: {:.4f}".format(loss_test.item()),
          "accuracy_test: {:.4f}".format(acc_test.item()))


def main():
    testing_mode = len(sys.argv)==2 and sys.argv[1] == "test"
    ogb_mode = len(sys.argv)==2 and sys.argv[1] == "ogb"
    
    if(testing_mode):
        print("Testing mode...")
        
        #load test data
        features, labels, adj = get_data("test/testdata/test.content",
                                         "test/testdata/test.cites")
    elif(ogb_mode):
        print("Benchmarking on OGB...")
        
        #load ogb data
        features, labels, adj = get_data('arxiv',
                                         'arxiv',
                                         directed = True)
    else:
        #load config data
        data_configs = json.load(open("config/data-params.json"))
        print(data_configs)

        #load cora data
        features, labels, adj = get_data(data_configs["feature_address"],
                                         data_configs["edges_address"],
                                         data_configs["encoding"],
                                         data_configs["directed"])

    model_configs = json.load(open("config/model-params.json"))
    print(model_configs)

    #train and test all the models and report losses and accuracy
    num_epochs = model_configs["num_epochs"]
    learning_rate = model_configs["learning_rate"]
    num_hidden = model_configs["num_hidden"]
    Q = int(model_configs["Q"])
    K = int(model_configs["K"])

    #initialize models
    in_features = list(features.size())[0]
    in_features_1 = list(features.size())[1]
    num_classes = len(set(labels))
    models = [Fully1Net(in_features, num_classes),
              Fully2Net(in_features, num_hidden, num_classes),
              Graph1Net(in_features_1, num_hidden, num_classes),
              Graph2Net(in_features_1, num_hidden, num_classes)]

    #split for train and test sets
    idx_train = torch.LongTensor(range(int(in_features_1 * 0.69)))
    idx_test = torch.LongTensor(range(int(in_features_1 * 0.69), in_features_1))

    #initialize optimizers
    ops = [optim.Adam(model.parameters(), lr=learning_rate) for model in models]

    model_names = ["Fully1Net", "Fully2Net", "Graph1Net", "Graph2Net"]
    for i in range(len(models)):
        print("\nRunning {} Model...".format(model_names[i]))
        for epoch in range(num_epochs):
            train_model(models[i], ops[i], epoch, idx_train, adj, features, labels)
        test_model(models[i], idx_test, adj, features, labels)

        
    #GraphSage
    feat_data = nn.Embedding(2708, 1433)
    feat_data.weight = nn.Parameter(torch.FloatTensor(features), requires_grad=False)

    agg1 = MeanAggregator(feat_data, cuda=True)
    enc1 = Encoder(feat_data, in_features_1, 128, adj, agg1, gcn=True, cuda=False)
    
    pooling = lambda nodes : enc1(nodes).t()
    
    agg2 = MeanAggregator(pooling, cuda=False)
    enc2 = Encoder(pooling, enc1.embed_dim, 128, adj, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = Q
    enc2.num_samples = Q

    #Mean GraphSage
    graphsage = SupervisedGraphSage(K, enc1)
    
    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    for epoch in range(num_epochs):
        loss = graphsage.loss(idx_train, 
                Variable(torch.LongTensor(labels[idx_train])))
        loss.backward()
        optimizer.step()
        print (epoch, loss.item())
    test_GraphSage(graphsage, idx_test, labels)
    
    #Pooling GraphSage
    graphsage = SupervisedGraphSage(K, enc2)
    
    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    for epoch in range(num_epochs):
        loss = graphsage.loss(idx_train, 
                Variable(torch.LongTensor(labels[idx_train])))
        loss.backward()
        optimizer.step()
        print (epoch, loss.item())
    test_GraphSage(graphsage, idx_test, labels)
    
if __name__ == '__main__':
    main()
