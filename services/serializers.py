from rest_framework import serializers


class UserSerializer(serializers.Serializer):
    uid = serializers.CharField()
    email = serializers.EmailField(allow_null=True, required=False)
    name = serializers.CharField(allow_null=True, required=False)


class PetSerializer(serializers.Serializer):
    id = serializers.CharField(required=False)
    name = serializers.CharField()
    gender = serializers.ChoiceField(choices=['male', 'female'])
    species = serializers.ChoiceField(choices=['dog', 'cat'])
    breed = serializers.CharField(allow_blank=True, required=False)
    dateOfBirth = serializers.DateField(required=False)
    photoUrl = serializers.URLField(allow_blank=True, required=False)


class EmotionScanRequestSerializer(serializers.Serializer):
    petId = serializers.CharField()
    mediaType = serializers.ChoiceField(choices=['image', 'audio'])
    file = serializers.FileField()


class EmotionScanResponseSerializer(serializers.Serializer):
    emotion = serializers.ChoiceField(choices=['happy', 'sad', 'anxious', 'excited', 'neutral', 'calm', 'aggressive', 'playful', 'sleepy'])
    confidence = serializers.FloatField()
    mediaUrl = serializers.URLField(allow_blank=True, required=False)
    petId = serializers.CharField()
    animalType = serializers.CharField(required=False)
    analysisMethod = serializers.CharField(required=False)
    topEmotions = serializers.ListField(required=False)
    aiDetectorType = serializers.CharField(required=False)


class EmotionLogSerializer(serializers.Serializer):
    id = serializers.CharField(required=False)
    petId = serializers.CharField()
    timestamp = serializers.DateTimeField()
    emotion = serializers.CharField()
    confidence = serializers.FloatField(allow_null=True, required=False)
    mediaUrl = serializers.URLField(allow_blank=True, required=False)


class RegisterSerializer(serializers.Serializer):
    name = serializers.CharField()
    email = serializers.EmailField()
    number = serializers.CharField()
    password = serializers.CharField(min_length=6, write_only=True)
    confirmPassword = serializers.CharField(min_length=6, write_only=True)

    def validate(self, attrs):
        if attrs['password'] != attrs['confirmPassword']:
            raise serializers.ValidationError('Passwords do not match')
        return attrs


class SendOtpSerializer(serializers.Serializer):
    uid = serializers.CharField()


class VerifyOtpSerializer(serializers.Serializer):
    uid = serializers.CharField()
    code = serializers.CharField()


class VerifyOtpRegistrationSerializer(serializers.Serializer):
    phoneNumber = serializers.CharField()
    code = serializers.CharField()


class ResendOtpRegistrationSerializer(serializers.Serializer):
    phoneNumber = serializers.CharField()


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()


class ForgotPasswordSerializer(serializers.Serializer):
    phoneNumber = serializers.CharField(help_text="Phone number to send OTP for password reset")


class VerifyResetOtpSerializer(serializers.Serializer):
    phoneNumber = serializers.CharField(help_text="Phone number used for password reset")
    code = serializers.CharField(help_text="OTP code received via SMS")


class ResetPasswordSerializer(serializers.Serializer):
    phoneNumber = serializers.CharField(help_text="Phone number used for password reset")
    password = serializers.CharField(min_length=6, write_only=True, help_text="New password")
    confirmPassword = serializers.CharField(min_length=6, write_only=True, help_text="Confirm new password")
    
    def validate(self, attrs):
        if attrs['password'] != attrs['confirmPassword']:
            raise serializers.ValidationError('Passwords do not match')
        return attrs


class UpdateProfileSerializer(serializers.Serializer):
    name = serializers.CharField(required=False, allow_blank=True)
    number = serializers.CharField(required=False, allow_blank=True)
    location = serializers.CharField(required=False, allow_blank=True)
    photoUrl = serializers.URLField(required=False, allow_blank=True)


class ChangePasswordSerializer(serializers.Serializer):
    currentPassword = serializers.CharField(write_only=True)
    newPassword = serializers.CharField(min_length=6, write_only=True)
    confirmPassword = serializers.CharField(min_length=6, write_only=True)

    def validate(self, attrs):
        if attrs['newPassword'] != attrs['confirmPassword']:
            raise serializers.ValidationError('Passwords do not match')
        if attrs['newPassword'] == attrs['currentPassword']:
            raise serializers.ValidationError('New password must be different from current password')
        return attrs


class GoogleSignInSerializer(serializers.Serializer):
    idToken = serializers.CharField(help_text="Google ID token from mobile SDK")
    name = serializers.CharField(required=False, allow_blank=True, help_text="User's display name")
    email = serializers.EmailField(required=False, allow_blank=True, help_text="User's email address")
    phoneNumber = serializers.CharField(required=False, allow_blank=True, help_text="User's phone number")


class AppleSignInSerializer(serializers.Serializer):
    identityToken = serializers.CharField(help_text="Apple identity token from mobile SDK")
    authorizationCode = serializers.CharField(required=False, allow_blank=True, help_text="Apple authorization code")
    name = serializers.CharField(required=False, allow_blank=True, help_text="User's display name")
    email = serializers.EmailField(required=False, allow_blank=True, help_text="User's email address")
    phoneNumber = serializers.CharField(required=False, allow_blank=True, help_text="User's phone number")


# Community/Posts Serializers
class PostSerializer(serializers.Serializer):
    id = serializers.CharField(required=False)
    content = serializers.CharField(help_text="Post content/text")
    images = serializers.ListField(
        child=serializers.URLField(),
        required=False,
        allow_empty=True,
        help_text="List of image URLs"
    )
    tags = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        allow_empty=True,
        help_text="List of hashtags/tags"
    )
    isPublic = serializers.BooleanField(default=True, help_text="Whether post is public")
    authorId = serializers.CharField(required=False)
    authorName = serializers.CharField(required=False)
    createdAt = serializers.DateTimeField(required=False)
    updatedAt = serializers.DateTimeField(required=False)
    likesCount = serializers.IntegerField(required=False, default=0)
    commentsCount = serializers.IntegerField(required=False, default=0)
    sharesCount = serializers.IntegerField(required=False, default=0)
    isLikedByUser = serializers.BooleanField(required=False, default=False)


class CreatePostSerializer(serializers.Serializer):
    content = serializers.CharField(help_text="Post content/text")
    tags = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        allow_empty=True,
        help_text="List of hashtags/tags"
    )
    isPublic = serializers.BooleanField(default=True, help_text="Whether post is public")


class CommentSerializer(serializers.Serializer):
    id = serializers.CharField(required=False)
    postId = serializers.CharField(help_text="ID of the post being commented on")
    content = serializers.CharField(help_text="Comment content")
    authorId = serializers.CharField(required=False)
    authorName = serializers.CharField(required=False)
    createdAt = serializers.DateTimeField(required=False)
    updatedAt = serializers.DateTimeField(required=False)
    likesCount = serializers.IntegerField(required=False, default=0)
    isLikedByUser = serializers.BooleanField(required=False, default=False)


class CreateCommentSerializer(serializers.Serializer):
    postId = serializers.CharField(help_text="ID of the post being commented on")
    content = serializers.CharField(help_text="Comment content")


class LikeSerializer(serializers.Serializer):
    pass  # postId comes from URL path


class ShareSerializer(serializers.Serializer):
    message = serializers.CharField(required=False, allow_blank=True, help_text="Optional message when sharing")


# Support/Help Desk Serializers (Simplified)
class SupportMessageSerializer(serializers.Serializer):
    id = serializers.CharField(required=False)
    email = serializers.EmailField(help_text="User's email for support responses")
    details = serializers.CharField(help_text="Support message details")
    userId = serializers.CharField(required=False)
    status = serializers.ChoiceField(
        choices=[
            ('new', 'New'),
            ('read', 'Read'),
            ('replied', 'Replied'),
        ],
        default='new',
        required=False
    )
    createdAt = serializers.DateTimeField(required=False)
    updatedAt = serializers.DateTimeField(required=False)
    adminReply = serializers.CharField(required=False, allow_blank=True)


class CreateSupportMessageSerializer(serializers.Serializer):
    email = serializers.EmailField(help_text="User's email for support responses")
    details = serializers.CharField(help_text="Support message details")


