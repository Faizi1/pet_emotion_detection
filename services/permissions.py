from rest_framework.permissions import BasePermission


class IsAdminFirebase(BasePermission):
    def has_permission(self, request, view) -> bool:
        user = getattr(request, 'user', None)
        return bool(user and getattr(user, 'is_admin', False))


