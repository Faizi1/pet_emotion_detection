from rest_framework import serializers


class VerifyReceiptSerializer(serializers.Serializer):
    receipt_data = serializers.CharField(required=False, allow_blank=True)
    product_id = serializers.CharField()
    transaction_id = serializers.CharField()
    original_transaction_id = serializers.CharField(required=False, allow_blank=True)


