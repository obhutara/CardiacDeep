
## experimental to output all dice scores -- WORKS!
dataset_iter = iter(validation_loader2)
for i in range(10):
    try:
        inputs, targets = prepare_batch(next(dataset_iter), 16, device)  # 2nd argument: training_batch_size
    except StopIteration:
        dataset_iter = iter(validation_loader2)
        inputs, targets = prepare_batch(next(dataset_iter), 16, device)
    model.eval()
    with torch.no_grad():
        logits = forward(model, inputs)
    labels = logits.argmax(dim=CHANNELS_DIMENSION, keepdim=True)
    dice_score.append(get_dice_score(F.softmax(logits, dim=CHANNELS_DIMENSION), targets))
dice_score = torch.cat(dice_score)
dice_score = dice_score.cpu().numpy()
t = range(0,(dice_score.shape[0]))
plt.plot(t, dice_score[:,0],'k',t, dice_score[:,1],'r',t, dice_score[:,2],'b',
         t, dice_score[:,3],'y',t, dice_score[:,4],'c',t, dice_score[:,5],'m',
         t, dice_score[:,6],'g')
plt.show()
##

## to store all in one dice numpy and plot line
dicescorenumpy1 = np.load('dice_score1.npy')
dicescorenumpy2 = np.load('dice_score2.npy')
dicescorenumpy3 = np.load('dice_score3.npy')
dicescorenumpy4 = np.load('dice_score4.npy')
dicescorenumpy5 = np.load('dice_score5.npy')
dicescorenumpy6 = np.load('dice_score6.npy')
dicescorenumpy7 = np.load('dice_score7.npy')
dicescorenumpy8 = np.load('dice_score8.npy')
dicescorenumpy9 = np.load('dice_score9.npy')

#val
diceappended = np.append(dicescorenumpy1,dicescorenumpy2,0)
diceappended_val = np.append(diceappended,dicescorenumpy3,0)

np.save('dice_score_val.npy',diceappended_val)
#test
diceappended = np.append(dicescorenumpy4,dicescorenumpy5,0)
#diceappended = np.append(diceappended,dicescorenumpy4,0)
#diceappended = np.append(diceappended,dicescorenumpy5,0)
diceappended = np.append(diceappended,dicescorenumpy6,0)
diceappended = np.append(diceappended,dicescorenumpy7,0)
diceappended = np.append(diceappended,dicescorenumpy8,0)
dicescorenumpy9 = np.delete(dicescorenumpy9, slice(15),0)
diceappended_test = np.append(diceappended,dicescorenumpy9,0)


np.save('dice_score_test.npy',diceappended_test)

#t = range(0,(diceappended_val.shape[0]))
t = range(0,(diceappended_test.shape[0]))
plt.plot(t, diceappended[:,0],'k',t, diceappended[:,1],'r',t, diceappended[:,2],'b',
         t, diceappended[:,3],'y',t, diceappended[:,4],'g',t, diceappended[:,5],'c',
         t, diceappended[:,6],'m')
plt.show()




import matplotlib.pyplot as plt
background_score = diceappended[:,0]
lv_score = diceappended[:,1]
rv_score = diceappended[:,2]
la_score = diceappended[:,3]
ra_score = diceappended[:,4]
aorta_score = diceappended[:,5]
pa_score = diceappended[:,6]
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
background = ax.scatter(t, background_score, color='k', s=0.4)
lv = ax.scatter(t, lv_score, color='r', s=0.4)
rv = ax.scatter(t, rv_score, color='b', s=0.4)
la = ax.scatter(t, la_score, color='y', s=0.4)
ra = ax.scatter(t, ra_score, color='g', s=0.4)
aorta = ax.scatter(t, aorta_score, color='c', s=0.4)
pa = ax.scatter(t, pa_score, color='m', s=0.4)
ax.set_xlabel('Validation sample number')
ax.set_ylabel('Dice Score')
ax.set_title('Dice scores for each validation sample by each part of the cardiovascular system ')
# produce a legend with the unique colors from the scatter
plt.legend((background, lv, rv, la, ra, aorta, pa),
           ('Background', 'Left Ventricle', 'Right Ventricle', 'Left Atrium', 'Right Atrium', 'Aorta', 'Pulmonary artery'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=11)
plt.show()



##validation

import matplotlib.pyplot as plt
background_score = diceappended_val[:,0]
lv_score = diceappended_val[:,1]
rv_score = diceappended_val[:,2]
la_score = diceappended_val[:,3]
ra_score = diceappended_val[:,4]
aorta_score = diceappended_val[:,5]
pa_score = diceappended_val[:,6]
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
background = ax.scatter(t, background_score, color='k', s=0.4)
lv = ax.scatter(t, lv_score, color='r', s=0.4)
rv = ax.scatter(t, rv_score, color='b', s=0.4)
la = ax.scatter(t, la_score, color='g', s=0.4)
ra = ax.scatter(t, ra_score, color='y', s=0.4)
aorta = ax.scatter(t, aorta_score, color='c', s=0.4)
pa = ax.scatter(t, pa_score, color='m', s=0.4)
ax.set_xlabel('Validation sample number')
ax.set_ylabel('Dice Score')
ax.set_title('Dice scores for each validation sample by each part of the cardiovascular system ')
# produce a legend with the unique colors from the scatter
plt.legend((background, lv, rv, la, ra, aorta, pa),
           ('Background', 'Left Ventricle', 'Right Ventricle', 'Left Atrium', 'Right Atrium', 'Aorta', 'Pulmonary artery'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=11)
plt.show()



##test

import matplotlib.pyplot as plt
background_score = diceappended_test[:,0]
lv_score = diceappended_test[:,1]
rv_score = diceappended_test[:,2]
la_score = diceappended_test[:,3]
ra_score = diceappended_test[:,4]
aorta_score = diceappended_test[:,5]
pa_score = diceappended_test[:,6]
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
background = ax.scatter(t, background_score, color='k', s=0.4)
lv = ax.scatter(t, lv_score, color='r', s=0.4)
rv = ax.scatter(t, rv_score, color='b', s=0.4)
la = ax.scatter(t, la_score, color='g', s=0.4)
ra = ax.scatter(t, ra_score, color='y', s=0.4)
aorta = ax.scatter(t, aorta_score, color='c', s=0.4)
pa = ax.scatter(t, pa_score, color='m', s=0.4)
ax.set_xlabel('Test sample number')
ax.set_ylabel('Dice Score')
ax.set_title('Dice scores for each test sample by each part of the cardiovascular system ')
# produce a legend with the unique colors from the scatter
plt.legend((background, lv, rv, la, ra, aorta, pa),
           ('Background', 'Left Ventricle', 'Right Ventricle', 'Left Atrium', 'Right Atrium', 'Aorta', 'Pulmonary artery'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=11)
plt.show()


