import dataclasses
from enum import auto, Enum
from typing import List, Tuple
from PIL import Image
import io
import base64

VOCAB_IMAGE_W = 1000  # 224
VOCAB_IMAGE_H = 1000  # 224


from PIL import Image



def resize_image_to_shortest_edge(image, target_shortest_edge=336):
    """
    Resize the image such that the shortest edge is equal to the specified length,
    while maintaining the aspect ratio.
    
    Parameters:
    - image: PIL Image object
    - target_shortest_edge: The target length for the shortest edge

    Returns:
    - Resized PIL Image object
    """
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    if original_width < original_height:
        # 宽度是最短边
        new_width = target_shortest_edge
        new_height = int(target_shortest_edge / aspect_ratio)
    else:
        # 高度是最短边
        new_height = target_shortest_edge
        new_width = int(target_shortest_edge * aspect_ratio)
    
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image



class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False
    first_round: bool = True


    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    msg, image, image_process_mode = msg
                    if image_process_mode == "Pad":
                        def expand2square(pil_img, background_color=(122, 116, 104)):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image)
                    elif image_process_mode == "Crop":
                        pass
                    elif image_process_mode == "Raw+Processor":
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((336, 336))
                    else:
                        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")

                    if image_process_mode != "Raw+Processor":
                        max_hw, min_hw = max(image.size), min(image.size)
                        aspect_ratio = max_hw / min_hw
                        max_len, min_len = 800, 400
                        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                        longest_edge = int(shortest_edge * aspect_ratio)
                        W, H = image.size
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        image = image.resize((W, H))
                    print('Input Image Size:{}'.format(image.size))
                    image = resize_image_to_shortest_edge(image, 336)
                    print('Resize input image to shortest edge size:{}'.format(image.size))

                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self, pred_mask=None):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    msg, image, image_process_mode = msg
                    if image_process_mode != "Raw+Processor":
                        max_hw, min_hw = max(image.size), min(image.size)
                        aspect_ratio = max_hw / min_hw
                        max_len, min_len = 800, 400
                        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                        longest_edge = int(shortest_edge * aspect_ratio)
                        W, H = image.size
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        image = image.resize((W, H))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    ret.append([img_str, None])
                    msg = msg.replace('<image>', '').strip()
                    if len(msg) > 0:
                        ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg

        # replace < and > with &lt and &gt respectively
        for idx, item in enumerate(ret):
            if idx > 0:
                if item[0] is not None:
                    item[0] = item[0].replace('<', '&lt;')
                    item[0] = item[0].replace('>', '&gt;')
                if item[1] is not None:
                    item[1] = item[1].replace('<', '&lt;')
                    item[1] = item[1].replace('>', '&gt;')

        # show predicted mask        
        if pred_mask is not None:
            pred_mask = Image.fromarray(pred_mask.astype('uint8'), 'RGB')
            buffered = io.BytesIO()
            pred_mask.save(buffered, format="PNG")
            pred_mask = buffered.getvalue()
            pred_mask = base64.b64encode(pred_mask).decode('utf-8')
            pred_mask = f'<img src="data:image/png;base64,{pred_mask}" alt="predicted mask" />'
            # do not show the <SEG> string when showing predicted mask
            # if ret[-1][1] == '<SEG>':
            #     ret.pop(-1)
            ret.append([None, pred_mask])
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }



medplib_conv_vicuna_v1_original_system = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "Assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language. "
           "In images, points are represented by coordinates [x, y]. The top-left corner is [0, 0]. The bottom-right corner is [width-1, height-1]. "
           "Increasing x moves right across the image while increasing y moves down. "
           "A bounding box is marked by [x1, y1, x2, y2] with the top-left and bottom-right points being [x1, y1] and [x2, y2] respectively. "
           f"The image size is assumed to be ({VOCAB_IMAGE_W}, {VOCAB_IMAGE_H}), i.e., width={VOCAB_IMAGE_W}, height={VOCAB_IMAGE_H}. "
           "Follow the instructions carefully. ",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

# medplib_conv_vicuna_v1 = Conversation(
#     system="A chat between a human and an AI that understands visuals. "
#            "In images, [x, y] denotes points: top-left [0, 0], bottom-right [width-1, height-1]. "
#            "Increasing x moves right; y moves down. "
#            f"Bounding box: [x1, y1, x2, y2]. Image size: {VOCAB_IMAGE_W}x{VOCAB_IMAGE_H}. "
#            "Follow instructions. ",
#     roles=("USER", "ASSISTANT"),
#     version="v1",
#     messages=(),
#     offset=0,
#     sep_style=SeparatorStyle.TWO,
#     sep=" ",
#     sep2="</s>",
# )
medplib_conv_vicuna_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)


default_conversation = medplib_conv_vicuna_v1
conv_templates = {
    "v1": medplib_conv_vicuna_v1,
    "medplib_v1": medplib_conv_vicuna_v1,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
