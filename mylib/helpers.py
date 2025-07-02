from dataclasses import dataclass, asdict, replace, fields
import itertools

@dataclass(frozen=True)
class Config:
    def to_dict(self):
        return asdict(self)

    def __str__(self):
        lines = []
        max_key_len = max(len(k) for k in self.__dataclass_fields__)

        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if value is None:
                continue
            formatted_value = (
                f"[{', '.join(map(str, value))}]" if isinstance(value, list)
                else repr(value)
            )
            lines.append(f"{key:<{max_key_len}} = {formatted_value}")
        return "\n".join(lines)
    
    def __call__(self, **kwargs) -> 'Config':
        """  eturn a new instance of this Config with specified fields replaced."""
        invalid  = set(kwargs) - set(self.__dataclass_fields__)
        if invalid:
            raise AttributeError(f"Unknown fields for {type(self).__name__}: {invalid}")
        return replace(self, **kwargs)

    def split(self):
        # Split fields based on whether they contains lists
        fields_list = {}
        fields_no_list = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Config):
                fields_list[field.name] = value.split()
            elif isinstance(value, list):
                # Check if value in list is a config
                flat = []
                for elem in value:
                    if isinstance(elem, Config):
                        flat.extend(elem.split())
                    else:
                        flat.append(elem)
                fields_list[field.name] = flat

            else:
                fields_no_list[field.name] = value

        # Make every combination of values and make new configs
        combinations = [dict(zip(fields_list.keys(), vals)) for vals in itertools.product(*fields_list.values())]
        cnfgs_split = []
        for combo in combinations:
            cnfgs_split.append(type(self)(**combo, **fields_no_list))

        return cnfgs_split 

    @property
    def shape_combination(self) -> tuple[int]:

        def collect_dims(cfg) -> list[int]:
            dims = []
            for field in fields(cfg):
                val = getattr(cfg, field.name)
                if isinstance(val, list):
                    dims.append(len(val))
                elif isinstance(val, Config):
                    dims.extend(collect_dims(val))
            return dims

        return tuple(collect_dims(self))
