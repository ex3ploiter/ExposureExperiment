def auc_softmax_adversarial(model, test_loader, test_attack, epoch):
    soft = torch.nn.Softmax(dim=1)
    anomaly_scores = []
    test_labels = []
    print('AUC Adversarial Softmax Started ...')
    with tqdm(test_loader, unit="batch") as tepoch:
        torch.cuda.empty_cache()
        for i, (data, target) in enumerate(tepoch):
            data, target = data.to(device), target.to(device)
            labels = target.to(device)
            adv_data = test_attack(data, target)
            output = model(adv_data)
            probs = soft(output).squeeze()
            anomaly_scores += probs[:, 1].detach().cpu().numpy().tolist()
            test_labels += target.detach().cpu().numpy().tolist()

    auc = roc_auc_score(test_labels, anomaly_scores)
    print(f'AUC Adversairal - Softmax - score on epoch {epoch} is: {auc * 100}')
    return auc

def auc_softmax(model, test_loader, epoch):
    model.eval()
    soft = torch.nn.Softmax(dim=1)
    anomaly_scores = []
    test_labels = []
    print('AUC Softmax Started ...')
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            torch.cuda.empty_cache()
            for i, (data, target) in enumerate(tepoch):
                data, target = data.to(device), target.to(device)
                output = model(data)
                probs = soft(output).squeeze()
                anomaly_scores += probs[:, 1].detach().cpu().numpy().tolist()
                test_labels += target.detach().cpu().numpy().tolist()

    auc = roc_auc_score(test_labels, anomaly_scores)
    print(f'AUC - Softmax - score on epoch {epoch} is: {auc * 100}')

    return auc