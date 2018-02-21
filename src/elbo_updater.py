
import chainer
import chainer.functions as F

from chainer import Variable
from chainer import training

class ELBOUpdater(training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.elbo, self.p_obj = kwargs.pop('models')
        self.encode = self.elbo.encode
        self.decode = self.p_obj.decode

        super(ELBOUpdater, self).__init__(*args, **kwargs)

    def compute_elbo(self, elbo, x):
        obj = elbo(x)
        chainer.report({'elbo': obj}, elbo)
        return obj

    def compute_pobj(self, p_obj, x):
        obj = p_obj(x)
        chainer.report({'obj': obj}, p_obj)
        return obj

    def update_core(self):
        elbo_optimizer = self.get_optimizer('elbo')
        pobj_optimizer = self.get_optimizer('p_obj')

        batch = self.get_iterator('main').next()
        x = Variable(self.converter(batch, self.device))
        xp = chainer.cuda.get_array_module(x.data)

        elbo, p_obj = self.elbo, self.p_obj

        # Update q, hold p fixed
        self.encode.enable_update()
        self.decode.disable_update()
        elbo.cleargrads()
        elbo_optimizer.update(self.compute_elbo, elbo, x)

        # Fix q, update p
        self.encode.disable_update()
        self.decode.enable_update()
        p_obj.cleargrads()
        pobj_optimizer.update(self.compute_pobj, p_obj, x)

