# Firebase Storage Setup Guide

## Overview

Firebase Storage is **already implemented** in your code! It's used for uploading:
- Pet photos
- Emotion scan images
- Emotion scan audio files
- Community post images

The code automatically falls back to local storage if Firebase Storage is not configured.

## Current Status

Based on your `petmood-firebase.json`, your project ID is: **`petmood-50a32`**

Your storage bucket should be: **`petmood-50a32.appspot.com`** (default) or a custom bucket name if you created one.

## How to Find Your Firebase Storage Bucket

### Method 1: Firebase Console (Recommended)

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project: **petmood-50a32**
3. Click on **Storage** in the left sidebar (or go to: https://console.firebase.google.com/project/petmood-50a32/storage)
4. At the top of the Storage page, you'll see your bucket name. It will look like:
   - `petmood-50a32.appspot.com` (default)
   - Or a custom name like `petmood-50a32-storage` (if you customized it)

### Method 2: Firebase Console Settings

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click the gear icon ⚙️ next to "Project Overview"
3. Select **Project Settings**
4. Scroll down to **Your apps** section
5. Look at any app configuration - the storage bucket is usually visible there
6. Or go to the **Storage** tab in settings

### Method 3: Check via Google Cloud Console

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select project: **petmood-50a32**
3. Go to **Cloud Storage** > **Buckets**
4. You'll see all your storage buckets listed

## Setup Instructions

### Step 1: Enable Firebase Storage (if not already enabled)

1. Go to [Firebase Console Storage](https://console.firebase.google.com/project/petmood-50a32/storage)
2. If you see "Get started", click it
3. Choose your storage location (preferably close to your users)
4. Start in **production mode** (for better security)
5. Click **Done**

### Step 2: Set Storage Rules (Important for Public Access)

Your code uses `blob.make_public()` to make files publicly accessible. Make sure your Storage rules allow this:

1. Go to Firebase Console > Storage > Rules
2. Set rules to allow read access:
   ```javascript
   rules_version = '2';
   service firebase.storage {
     match /b/{bucket}/o {
       match /{allPaths=**} {
         allow read: if true;  // Allow public reads
         allow write: if request.auth != null;  // Only authenticated users can write
       }
     }
   }
   ```
3. Click **Publish**

### Step 3: Configure Environment Variable

#### For Local Development (.env file)

Create a `.env` file in your project root with:

```env
FIREBASE_STORAGE_BUCKET=petmood-50a32.appspot.com
FIREBASE_PROJECT_ID=petmood-50a32
FIREBASE_CREDENTIALS_PATH=D:\Projects\Fiverr\pet_emotion_detection\petmood-firebase.json
```

**Important:** Replace `petmood-50a32.appspot.com` with your actual bucket name if it's different!

#### For Production (Render/Server)

Set these environment variables:
- `FIREBASE_STORAGE_BUCKET=petmood-50a32.appspot.com`
- `FIREBASE_PROJECT_ID=petmood-50a32`
- `FIREBASE_CREDENTIALS_JSON=<your-full-json-credentials>`

### Step 4: Verify Configuration

You can test if Firebase Storage is configured by calling the admin endpoint:

```bash
GET /api/admin/storage-config
```

This will return:
```json
{
  "storageConfigured": true,
  "bucketName": "petmood-50a32.appspot.com",
  "projectId": "petmood-50a32"
}
```

## How It Works

### Current Implementation

The code in `services/storage.py`:

1. **First tries Firebase Storage** - Uploads files to Firebase Storage bucket
2. **Falls back to local storage** - If Firebase Storage fails, saves to `media/` folder
3. **Returns public URL** - Either Firebase Storage URL or local media URL

### Code Flow

1. User uploads file via `/api/scans` or `/api/pets` endpoint
2. `upload_bytes()` function in `services/storage.py` is called
3. It attempts to upload to Firebase Storage using `get_bucket()` from `services/firebase.py`
4. If successful, returns Firebase Storage public URL
5. If failed, saves locally and returns local URL

## Troubleshooting

### Issue: "Firebase Storage is not configured"

**Solution:** Set `FIREBASE_STORAGE_BUCKET` environment variable

### Issue: Upload fails with permission error

**Solutions:**
1. Check that Storage is enabled in Firebase Console
2. Verify your service account has Storage Admin role in Google Cloud Console
3. Check Storage rules allow writes for authenticated users

### Issue: Files uploaded but not accessible

**Solutions:**
1. Ensure Storage rules allow public reads (see Step 2 above)
2. Check that `blob.make_public()` is working (code already does this)
3. Verify bucket name is correct

### Issue: Using custom bucket name

If you created a custom storage bucket:
- Update `FIREBASE_STORAGE_BUCKET` with the custom name
- Make sure it's in the format: `your-custom-bucket-name.appspot.com` or just `your-custom-bucket-name`

## Security Notes

1. **Service Account Permissions**: Your service account needs:
   - `Storage Admin` role (full access) OR
   - `Storage Object Admin` role (can create/delete objects)

2. **Storage Rules**: The rules above allow:
   - **Read**: Anyone (public)
   - **Write**: Only authenticated users

3. **Alternative Rules** (more secure):
   ```javascript
   rules_version = '2';
   service firebase.storage {
     match /b/{bucket}/o {
       match /{userId}/{allPaths=**} {
         allow read: if true;
         allow write: if request.auth != null && request.auth.uid == userId;
       }
     }
   }
   ```

## Testing

After setup, test by:

1. **Upload a pet photo:**
   ```bash
   POST /api/pets
   Content-Type: multipart/form-data
   
   {
     "name": "Test Pet",
     "gender": "male",
     "species": "dog",
     "photo": <file>
   }
   ```

2. **Check response** - Should contain `photoUrl` pointing to Firebase Storage URL:
   ```
   https://firebasestorage.googleapis.com/v0/b/petmood-50a32.appspot.com/o/...
   ```

3. **Verify file in Firebase Console** - Go to Storage and see your uploaded file

## Additional Resources

- [Firebase Storage Documentation](https://firebase.google.com/docs/storage)
- [Storage Security Rules](https://firebase.google.com/docs/storage/security)
- [Python Firebase Admin SDK](https://firebase.google.com/docs/admin/setup)

