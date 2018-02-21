
import chainer
import chainer.functions as F
from chainer import Variable

import model

class ISVAEUpdater(chainer.training.StandardUpdater):
    def __init__(self, num_zsamples=8, *args, **kwargs):
        self.encoder, self.decoder = kwargs.pop('models')
        self.num_zsamples = num_zsamples
        super(ISVAEUpdater, self).__init__(*args, **kwargs)
        self.elbo = model.ELBOObjective(self.encoder, self.decoder, self.num_zsamples)
        self.isobjective = model.ISObjective(self.encoder, self.decoder, self.num_zsamples)

    def encoder_objective(self, encoder, x):
        obj_elbo = self.elbo(x)
        chainer.report({'elbo': obj_elbo}, encoder)
        return obj_elbo

    def decoder_objective(self, decoder, x):
        obj_is = self.isobjective(x)
        chainer.report({'is': obj_is}, decoder)
        return obj_is

    def update_core(self):
        batch = self.get_iterator('main').next()
        x = Variable(self.converter(batch, self.device))
        xp = chainer.cuda.get_array_module(x.data)

        self.encoder.cleargrads()
        enc_optimizer = self.get_optimizer('encoder')
        enc_optimizer.update(self.encoder_objective, self.encoder, x)

        self.decoder.cleargrads()
        dec_optimizer = self.get_optimizer('decoder')
        dec_optimizer.update(self.decoder_objective, self.decoder, x)

