import logging
import requests
import traceback

from one.api import ONE

_logger = logging.getLogger('ibllib')


class Snapshot:
    """
    A class to register images in form of Notes, linked to an object on Alyx.

    :param object_id: The id of the object the image should be linked to
    :param content_type: Which type of object to link to, e.g. 'session', 'probeinsertions', 'subject',
    default is 'session'
    :param one: An ONE instance, if None is given it will be instantiated.
    """

    def __init__(self, object_id, content_type='session', one=None):
        self.one = one or ONE()
        self.object_id = object_id
        self.content_type = content_type
        self.images = []

    def generate_image(self, plt_func, plt_kwargs):
        """
        Takes a plotting function and adds the output to the Snapshot.images list for registration

        :param plt_func: A plotting function that returns the path to an image.
        :param plt_kwargs: Dictionary with keyword arguments for the plotting function
        """
        img_path = plt_func(**plt_kwargs)
        self.images.append(img_path)
        return img_path

    def register_image(self, image_file, text='', width=None):
        """
        Registers an image as a Note, attached to the object specified by Snapshot.object_id

        :param image_file: Path to the image to to registered
        :param text: Text to describe the image, defaults ot empty string
        :param width: width to scale the image to, defaults to None (scale to UPLOADED_IMAGE_WIDTH in alyx.settings.py),
        other options are 'orig' (don't change size) or any integer (scale to width=int, aspect ratios won't be changed)

        :returns: dict, note as registered in database
        """
        fig_open = open(image_file, 'rb')
        note = {
            'user': self.one.alyx.user, 'content_type': self.content_type, 'object_id': self.object_id,
            'text': text, 'width': width}
        _logger.info(f'Registering image to {self.content_type} with id {self.object_id}')
        # Catch error that results from object_id - content_type mismatch
        try:
            note_db = self.one.alyx.rest('notes', 'create', data=note, files={'image': fig_open})
            fig_open.close()
            return note_db
        except requests.HTTPError as e:
            if "matching query does not exist.'" in str(e):
                fig_open.close()
                _logger.error(f'The object_id {self.object_id} does not match an object of type {self.content_type}')
                _logger.debug(traceback.format_exc())
            else:
                fig_open.close()
                raise

    def register_images(self, image_list=None, texts=[''], widths=[None]):
        """
        Registers a list of images as Notes, attached to the object specified by Snapshot.object_id.
        The images can be passed as image_list. If None are passed, will try to register the images in Snapshot.images.

        :param image_list: List of paths to the images to to registered. If None, will try to register any images in
                           Snapshot.images
        :param texts: List of text to describe the images. If len(texts)==1, the same text will be used for all images
        :param widths: List of width to scale the figure to (see Snapshot.register_image). If len(widths)==1,
                       the same width will be used for all images

        :returns: list of dicts, notes as registered in database
        """
        if not image_list or len(image_list) == 0:
            if len(self.images) == 0:
                _logger.warning(
                    "No figures were passed to register_figures, and self.figures is empty. No figures to register")
                return
            else:
                image_list = self.images
        if len(texts) == 1:
            texts = len(image_list) * texts
        if len(widths) == 1:
            widths = len(image_list) * widths
        note_dbs = []
        for figure, text, width in zip(image_list, texts, widths):
            note_dbs.append(self.register_image(figure, text=text, width=width))
        return note_dbs
