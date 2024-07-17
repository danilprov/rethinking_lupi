from PIL import Image
from keras import Input, Model
import numpy as np
from keras.src.layers import Dense, Activation
from keras.src.losses import categorical_crossentropy
from keras.src.optimizers import RMSprop
from keras.callbacks import EarlyStopping


def downsample(x, p_down):
  s_tr = []

  for i in range(x.shape[0]):
    img = Image.fromarray(x[i].reshape(28, 28))
    new_size = (int(img.width * p_down), int(img.height * p_down))
    img_resized = img.resize(new_size, resample=Image.BILINEAR)
    s_tr.append(np.array(img_resized).ravel())

  # Convert the list of arrays to a 2D numpy array
  s_tr = np.array(s_tr)
  return s_tr


def MLP(d, m, q):
  input_layer = Input(shape=(d,))
  x = Dense(m, activation='relu')(input_layer)
  x = Dense(m, activation='relu')(x)
  pre_softmax = Dense(q, name="pre_softmax")(x)  # Output before softmax
  output = Activation('softmax', name="softmax")(pre_softmax)  # Output after softmax

  # RMSprop(clipvalue=1.0)

  model = Model(inputs=input_layer, outputs=[output, pre_softmax])
  # model.compile(optimizer=RMSprop(clipvalue=1.0),
  #               loss={'softmax': 'categorical_crossentropy', 'pre_softmax': None})  # loss='categorical_crossentropy')
  model.compile(optimizer='rmsprop', loss={'softmax': 'categorical_crossentropy', 'pre_softmax': None})
  # model.compile(optimizer='RMSprop', loss='categorical_crossentropy')

  return model


def softmax(w, t = 1.0):
    e = np.exp(w / t)
    return e/np.sum(e,1)[:,np.newaxis]


def weighted_loss(base_loss,l):
    def loss_function(y_true, y_pred):
        return l*base_loss(y_true,y_pred)
    return loss_function


def distillation(d, m, q, t, l):
    input_layer = Input(shape=(d,), name='input_layer')
    x = Dense(m, activation='relu')(input_layer)
    x = Dense(m, activation='relu')(x)
    pre_softmax = Dense(q)(x)

    hard_softmax = Activation('softmax', name='hard_softmax')(pre_softmax)
    soft_softmax = Activation('softmax', name='soft_softmax')(pre_softmax)

    loss_hard = weighted_loss(categorical_crossentropy, 1. - l)
    loss_soft = weighted_loss(categorical_crossentropy, t * t * l)

    model = Model(inputs=input_layer, outputs=[hard_softmax, soft_softmax])
    model.compile(optimizer='rmsprop', loss={'hard_softmax': loss_hard, 'soft_softmax': loss_soft})
    return model


def load_data(dataset):
  d = np.load('./data/' + dataset + '.npz','r')
  x_tr = d['x_train'].astype(np.float32)
  x_te = d['x_test'].astype(np.float32)
  y_tr = to_one_hot(d['y_train'].astype(np.float32))
  y_te = to_one_hot(d['y_test'].astype(np.float32))
  return x_tr, y_tr, x_te, y_te


def to_one_hot(array):
  num_classes = np.max(array) + 1
  return np.eye(int(num_classes))[array.astype(int)]


if __name__ == "__main__":
  np.random.seed(0)

  p_downsample = .25
  N_samples = 10000#int(sys.argv[1])
  hidden_size = 20
  max_epochs = 400
  batch_size = 1000
  outfile = open('./logs/result_mnist_varying_size_epochs', 'a')
  outfile.write('epochs, sample_size, downscale_percentage, temperature, lambda, teacher, no-pi, student, self_dist_student\n')
  early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=0)
  results = []
  for max_epochs in [30, 50, 100, 200, 300, 400]:
    for N_samples in [300, 500]:#, 1000, 5000, 10000, 20000]:
      ax_tr, ay_tr, x_te, y_te = load_data('mnist')

      xs_te = downsample(x_te,p_downsample)
      x_te = x_te.reshape(x_te.shape[0], x_te.shape[1] * x_te.shape[2])
      x_te  = x_te/255.0
      xs_te = xs_te/255.0

      for rep in range(10):
        # random training split
        i     = np.random.permutation(ax_tr.shape[0])[0:N_samples]
        x_tr  = ax_tr[i]
        y_tr  = ay_tr[i]
        xs_tr = downsample(x_tr,p_downsample)
        x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1] * x_tr.shape[2])
        x_tr  = x_tr/255.0
        xs_tr = xs_tr/255.0

        # big mlp
        mlp_big = MLP(x_tr.shape[1], hidden_size, y_tr.shape[1])
        mlp_big.fit(x_tr, y_tr, epochs=max_epochs, verbose=0)
        predictions = mlp_big.predict(x_te, verbose=0)[0]
        predicted_classes = np.argmax(predictions, axis=1)
        err_big = np.mean(predicted_classes == np.argmax(y_te, axis=1))
        std_big = np.std(predictions, axis=1).mean()

        teacher_self_dist = MLP(xs_tr.shape[1], hidden_size, y_tr.shape[1])
        teacher_self_dist.fit(xs_tr, y_tr, epochs=max_epochs, verbose=0)
        predictions = teacher_self_dist.predict(xs_te, verbose=0)[0]
        predicted_classes = np.argmax(predictions, axis=1)
        err_self_dist = np.mean(predicted_classes == np.argmax(y_te, axis=1))
        std_self_dist = np.std(predictions, axis=1).mean()

        # student mlp
        for t in [10]:  # "[1,2,5,10,20,50]:
          for L in [.5]:  # [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            try:
              # Privileged distillation
              pre_softmax_output = mlp_big.predict(x_tr)[1]  # [1] for the pre-softmax output
              ys_tr = softmax(pre_softmax_output, t)

              mlp_student = distillation(xs_tr.shape[1], hidden_size, ys_tr.shape[1], t, L)
              mlp_student.fit({'input_layer': xs_tr}, {'hard_softmax': y_tr, 'soft_softmax': ys_tr}, epochs=max_epochs, verbose=0)

              # Predict and evaluate error
              predictions = mlp_student.predict(xs_te)
              predicted_classes_hard = np.argmax(predictions[0], axis=1)
              actual_classes = np.argmax(y_te, axis=1)

              err_student = np.mean(predicted_classes_hard == actual_classes)

              # Self distillation
              pre_softmax_output = teacher_self_dist.predict(xs_tr)[1]  # [1] for the pre-softmax output
              ys_tr = softmax(pre_softmax_output, t)

              mlp_self_dist_student = distillation(xs_tr.shape[1], hidden_size, ys_tr.shape[1], t, L)
              mlp_self_dist_student.fit({'input_layer': xs_tr}, {'hard_softmax': y_tr, 'soft_softmax': ys_tr}, epochs=max_epochs,
                                        verbose=0)

              # Predict and evaluate error
              predictions = mlp_self_dist_student.predict(xs_te)
              predicted_classes_hard = np.argmax(predictions[0], axis=1)
              actual_classes = np.argmax(y_te, axis=1)

              err_self_dist_student = np.mean(predicted_classes_hard == actual_classes)

              line = [max_epochs, N_samples, p_downsample, t, L, round(err_big, 3), round(err_self_dist, 3), round(err_student, 3),
                      round(err_self_dist_student, 3)]
              results.append(line)
              print(line)
              outfile.write(str(line) + '\n')
            except Exception as e:
              print(xs_tr.shape, y_tr.shape, ys_tr.shape)
              print(e)

  outfile.close()
