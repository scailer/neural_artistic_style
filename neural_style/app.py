# -*- coding: utf-8 -*-

import uuid
import logging
from PIL import Image
from datetime import datetime

import tornado.ioloop
import tornado.web
import tornado.httpserver
from tornado import gen
from tornado.options import define, options

from neural_style.neural_artistic_style import *  # noqa

logger = logging.getLogger(__name__)
define("port", default=8888, help="run on the given port", type=int)


def load_image(body):
    if not body:
        return None

    name = 'content/{}.png'.format(uuid.uuid4())
    with open(name, 'w') as f:
        f.write(body)

    img = Image.open(name)
    img.save(name)
    return name


@gen.coroutine
def get_net(layers, init_img, subject_img, style_img, subject_weights,
            style_weights, smoothness):

    net = StyleNetwork(layers, to_bc01(init_img), to_bc01(subject_img),
                       to_bc01(style_img), subject_weights, style_weights,
                       smoothness)

    raise gen.Return(net)


network = 'imagenet-vgg-verydeep-16.mat'
# network = 'imagenet-vgg-verydeep-19.mat'
pool_method = 'avg'
layers, pixel_mean = vgg_net(network, pool_method=pool_method)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("/home/ubuntu/neural_artistic_style/form.html")

    @gen.coroutine
    def post(self):
        t = datetime.now()
        # random_seed = None
        subject_ratio = float(self.get_argument("subject_ratio") or 2e-2)
        style_weights = [(0, 1), (2, 1), (4, 1), (8, 1), (12, 1)]
        subject_weights = [(9, 1)]
        smoothness = 5e-8
        learn_rate = float(self.get_argument("learn_rate") or 2.0)
        iterations = int(self.get_argument("iterations") or 10)
        animation = 'animation'
        init_noise = 0.0
        style = self.get_argument("style") or 'images/starry_night.jpg'
        subject = 'images/tuebingen.jpg'

        subject = load_image(self.request.files['userfile'][0]['body'])

        if 'styleimg' in self.request.files:
            custom_style = load_image(self.request.files['styleimg'][0]['body'])
            style = custom_style or style
        print('Style {}'.format(style))

        logger.debug('Load image {}'.format(datetime.now() - t))
        print('Load image {}'.format(datetime.now() - t))

        # logger.debug('Load net {}'.format(datetime.now() - t))
        # print('Load net {}'.format(datetime.now() - t))

        # Inputs
        style_img = imread(style) - pixel_mean
        subject_img = imread(subject) - pixel_mean
        init_img = subject_img
        noise = np.random.normal(size=init_img.shape, scale=np.std(init_img)*1e-1)
        init_img = init_img * (1 - init_noise) + noise * init_noise
        logger.debug('Init img {}'.format(datetime.now() - t))
        print('Init img {}'.format(datetime.now() - t))

        # Setup network
        subject_weights = weight_array(subject_weights) * subject_ratio
        style_weights = weight_array(style_weights)
        net = yield get_net(layers, init_img, subject_img, style_img,
                            subject_weights, style_weights, smoothness)
        logger.debug('Setup net {}'.format(datetime.now() - t))
        print('Setup net {}'.format(datetime.now() - t))

        # Repaint image
        def net_img():
            return to_rgb(net.image) + pixel_mean

        if not os.path.exists(animation):
            os.mkdir(animation)

        params = net.params
        learn_rule = dp.Adam(learn_rate=learn_rate)
        learn_rule_states = [learn_rule.init_state(p) for p in params]
        logger.debug('Prepared {}'.format(datetime.now() - t))
        print('Prepared {}'.format(datetime.now() - t))

        for i in range(iterations):
            print('Iter 1 {}'.format(datetime.now() - t))
            cost = np.mean(net.update())
            print('Itre 2 {}'.format(datetime.now() - t))
            for param, state in zip(params, learn_rule_states):
                learn_rule.step(param, state)
            logger.debug('Iteration: %i, cost: %.4f' % (i, cost))
            print('Iteration: %i, cost: %.4f' % (i, cost))

        logger.debug('Ready {}'.format(datetime.now() - t))
        print('Ready {}'.format(datetime.now() - t))

        name = 'content/{}.png'.format(uuid.uuid4())
        imsave(name, net_img())
        Image.open(name).save(name.replace('png', 'jpg'), 'JPEG')
        self.write(open(name.replace('png', 'jpg'), 'r').read())
        self.set_header("Content-type",  "image/png")


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])


def main():
    tornado.options.parse_command_line()
    app = make_app()
    server = tornado.httpserver.HTTPServer(app)
    server.listen(options.port)
    print('Run')
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main()
