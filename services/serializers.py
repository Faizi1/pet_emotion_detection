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
    emotion = serializers.ChoiceField(choices=['happy', 'sad', 'anxious', 'excited', 'neutral'])
    confidence = serializers.FloatField()
    mediaUrl = serializers.URLField(allow_blank=True, required=False)
    petId = serializers.CharField()


class EmotionLogSerializer(serializers.Serializer):
    id = serializers.CharField(required=False)
    petId = serializers.CharField()
    timestamp = serializers.DateTimeField()
    emotion = serializers.CharField()
    confidence = serializers.FloatField()
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


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()


class ForgotPasswordSerializer(serializers.Serializer):
    email = serializers.EmailField()


