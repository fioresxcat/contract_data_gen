import PIL.ImageFont as ImageFont

class Font:
    def __init__(self, font_scale=20) -> None:  
        self.font_scale = font_scale
        self.font_dict = dict(
            normal = ImageFont.truetype(
            "fonts/time_new_roman.ttf", size=self.font_scale),
            big = ImageFont.truetype(
                "fonts/time_new_roman.ttf", size=self.font_scale+5),
            small = ImageFont.truetype(
                "fonts/time_new_roman.ttf", size=self.font_scale-2),
            extra_big = ImageFont.truetype(
                "fonts/time_new_roman.ttf", size=self.font_scale+10),

            bold = ImageFont.truetype(
                "fonts/Times-New-Roman-Bold_44652.ttf", size=self.font_scale),
            italic = ImageFont.truetype(
                "fonts/Times-New-Roman-Italic_44665.ttf", size=self.font_scale),
            bold_italic = ImageFont.truetype(
                'fonts/Times-New-Roman-Bold-Italic_44651.ttf', self.font_scale+5),
            big_bold = ImageFont.truetype( 
                "fonts/Times-New-Roman-Bold_44652.ttf", size=self.font_scale + 3),
            extra_big_bold = ImageFont.truetype( \
                "fonts/Times-New-Roman-Bold_44652.ttf", size=self.font_scale + 25),
            extra_small = ImageFont.truetype(
                "fonts/time_new_roman.ttf", size=self.font_scale-7),
        )

    def get_font(self, font_type):
        assert font_type in self.font_dict, "font_type %s not found"%font_type
        return self.font_dict[font_type]

        
