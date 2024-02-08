def testAccuracy():
    net.eval()
    accuracy = 0.0
    total = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    x=[]
    y=[]
    np.array(x)
    np.array(y)
    # net=torch.load('')
    with torch.no_grad():
        for i, (images, labels) in enumerate(valid_loader, 0):
            #images, labels = dataiter.next()
            # 输入数据进行预测
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            predata = net(images)
            #predata, labedata = predict(net, valid_loader)
            loss=loss_func(predata, labels)
            loss += loss.item()
            x=np.append(x,[i[0] for i in labels.data.cpu().numpy()])
            y=np.append(y,np.array([i[0] for i in predata.data.cpu().numpy()]))
    preloss = np.mean(loss.data.cpu().numpy()) #preloss = loss/(i+1)
    r2 = 1 - np.mean((y - x) ** 2) / np.mean((x - x.mean()) ** 2)
    vmse = np.mean((y - x) ** 2)
    return(r2,preloss,vmse)

def train(num_epochs,device):
    import time
    start_time = time.process_time()
    best_accuracy = 0.0

    # Define your execution device
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    k=0
    predata=[]
    labeldata=[]
    mseloss=[]
    trainnum=[]
    epochout=[]
    epochV=[]
    vloss=[]
    vr2=[]
    # Convert model parameters and buffers to CPU or Cuda
    net.to(device)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0
        for i, (images, labels) in enumerate(train_loader, 0):
            #images, labels = dataiter.next()
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            prediction = net(images)
            #print(prediction,labels)
            #print('Prediction data is:',prediction.data.cpu().numpy()[0])
            #print('label data is: ',labels.data.cpu().numpy())
            # 计算预测值与真值误差，注意参数顺序问题
            # 第一个参数为预测值，第二个为真值
            loss = loss_func(prediction, labels)*100
            # 开始优化步骤
            # 每次开始优化前将梯度置为0
            optimizer.zero_grad()
            # 误差反向传播
            loss.backward()
            # 按照最小loss优化参数
            optimizer.step()
            running_loss += loss.item()

            if (i+1) % 10 == 0:
                # print every 1000 (twice per epoch)
                #print(f"epoch #{epoch+1} Iteration #{i+1} loss: {loss_value}")
                print('[Epoch: %d/%d]- %d/%d | loss: %.3f | predict-True: %5d - %5d' %
                      (epoch + 1,num_epochs, i + 1,len(train_loader), loss.item(), prediction.data.cpu().numpy()[1]*116,labels.data.cpu().numpy()[1]*116))
                mseloss.append(running_loss/10)
                trainnum.append(i+1)
                epochout.append(epoch+1)
                result=pd.DataFrame({'epoch':epochout,'trainnum':trainnum,'loss':mseloss})
                result.to_csv("./result/"+casename+"/"+casename+"-result.csv",index=False,sep=',')
                # Tensorboard
                writer.add_scalar('Training loss',running_loss / 10,epoch * len(train_loader) + i)
                # writer.add_figure('predictions vs. actuals',
                #                   plot_classes_preds(net, inputs, labels),
                #                   global_step=epoch * len(trainloader) + i)
                # zero the loss
                running_loss = 0.0

            predata.append(prediction.data.cpu().numpy()[0])
            labeldata.append(labels.data.cpu().numpy()[0])

        r2,validloss,vmse = testAccuracy()
        vloss.append(validloss)
        vr2.append(r2)
        epochV.append(epoch+1)
        writer.add_scalar('valid_loss',validloss,epoch+1)
        vresult=pd.DataFrame({'epoch':epochV,'vloss':vloss,'vMSE':vmse,'vr2':vr2})
        vresult.to_csv("./result/"+casename+"/"+casename+"-result-valid.csv",index=False,sep=',')
        print('[Epoch: %d/%d] ======>| Valid Loss: %.3f ,MSE: %.3f | R^2 is %.3f' % (epoch+1,num_epochs,validloss,vmse,r2))
        if validloss <= 0.005:
            k +=1
            print('Validloss K=',k)
            if k == 3 and r2>0.90:
                torch.save(net, casename+'-OK'+'.pt')

            if k == 3 and r2>0.95:
                print('Traing is end, Due to Validloss consecutively less than 0.005.')
                break
        else: k = 0


    print('Finished Training')
    ttime=(np.array(epochout)-1)*len(train_loader)+np.array(trainnum)
    plt.figure()
    plt.legend()
    plt.grid(True)
    plt.ylim(0,0.1)
    plt.ylabel("Mean Squared Error")
    plt.xlabel("train times")
    plt.plot(ttime,mseloss)
    plt.savefig('tranloss.png', dpi=1200) 
    end_time = time.process_time()
    print("Use time:", (end_time-start_time)/3600)

def predict(model, device,dataloder):
    Vdataiter=iter(dataloder)
    vimg, vlabels = Vdataiter.next()
    model.to(device)
    with torch.no_grad():
        vimg=vimg.to(device)
        out = model(vimg)
        #_, pre = torch.max(out.data, 1)
        return out, vlabels