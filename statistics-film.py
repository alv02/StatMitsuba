import mitsuba as mi

mi.set_variant("llvm_ad_rgb")


class MyFilm(mi.Film):
    def __init__(self, props):
        super().__init__(props)
