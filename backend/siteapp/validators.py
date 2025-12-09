import re
from django.core.exceptions import ValidationError
from django.utils.translation import gettext as _


class ComplexityValidator:
    """Require at least one uppercase, one lowercase, one digit, and one special character."""

    def validate(self, password, user=None):
        if not re.search(r"[A-Z]", password or ""):
            raise ValidationError(_("Password must include at least one uppercase letter."))
        if not re.search(r"[a-z]", password or ""):
            raise ValidationError(_("Password must include at least one lowercase letter."))
        if not re.search(r"\d", password or ""):
            raise ValidationError(_("Password must include at least one digit."))
        if not re.search(r"[^A-Za-z0-9]", password or ""):
            raise ValidationError(_("Password must include at least one special character."))

    def get_help_text(self):
        return _("Password must contain uppercase, lowercase, digit, and special character.")


