import json


class SignalJSONDecoder(json.JSONDecoder):
    def decode(self, s, *args, **kwargs):
        s = s.replace("'", '"')
        s = s.replace('\\"', '"')
        s = s.strip('"')
        s = s.replace('"{', "{").replace('}"', "}")

        return super().decode(s, *args, **kwargs)