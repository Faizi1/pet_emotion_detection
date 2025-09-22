from django.contrib import admin
from django.urls import path
from django.template.response import TemplateResponse
from django.contrib.admin import AdminSite
from django.utils.decorators import method_decorator
from django.contrib.admin.views.decorators import staff_member_required
from .firebase import get_firestore
from django.shortcuts import redirect
from django.views.decorators.http import require_http_methods


class FirestoreDashboardAdmin(AdminSite):
    site_header = 'Petmood Admin'

    @method_decorator(staff_member_required)
    def firestore_dashboard_view(self, request):
        db = get_firestore()
        users_ref = db.collection('users')
        total_users = len(list(users_ref.stream()))
        total_pets = 0
        total_scans = 0
        # This is simple and potentially slow. For prod, maintain counters.
        for user_doc in users_ref.stream():
            total_pets += len(list(users_ref.document(user_doc.id).collection('pets').stream()))
            total_scans += len(list(users_ref.document(user_doc.id).collection('emotion_logs').stream()))
        # Charts data
        from datetime import datetime, timedelta, timezone
        last_7 = [(datetime.now(timezone.utc) - timedelta(days=i)).date() for i in range(6, -1, -1)]
        scans_per_day = {d.isoformat(): 0 for d in last_7}
        emotion_counts = {k: 0 for k in ['happy', 'sad', 'anxious', 'excited', 'neutral']}
        # Prefer collection group with ordering; if missing index, fallback per-user (slower but works for MVP)
        try:
            logs_iter = db.collection_group('emotion_logs').limit(1000).stream()
            for d in logs_iter:
                item = d.to_dict() or {}
                ts = item.get('timestamp')
                emo = (item.get('emotion') or '').lower()
                if hasattr(ts, 'date'):
                    ds = ts.date().isoformat()
                    if ds in scans_per_day:
                        scans_per_day[ds] += 1
                if emo in emotion_counts:
                    emotion_counts[emo] += 1
        except Exception:
            # Fallback: iterate each user's logs limited
            for user_doc in users_ref.stream():
                for log in users_ref.document(user_doc.id).collection('emotion_logs').limit(200).stream():
                    item = log.to_dict() or {}
                    ts = item.get('timestamp')
                    emo = (item.get('emotion') or '').lower()
                    if hasattr(ts, 'date'):
                        ds = ts.date().isoformat()
                        if ds in scans_per_day:
                            scans_per_day[ds] += 1
                    if emo in emotion_counts:
                        emotion_counts[emo] += 1
        context = {
            **self.each_context(request),
            'total_users': total_users,
            'total_pets': total_pets,
            'total_scans': total_scans,
            'chart_days': list(scans_per_day.keys()),
            'chart_values': list(scans_per_day.values()),
            'emotion_labels': list(emotion_counts.keys()),
            'emotion_values': list(emotion_counts.values()),
        }
        return TemplateResponse(request, 'admin/firestore_dashboard.html', context)

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('', lambda request: redirect('dashboard/'), name='index'),
            path('dashboard/', self.firestore_dashboard_view, name='dashboard'),
            path('users/', self.firestore_users_view, name='users'),
            path('users/<str:uid>/edit/', self.firestore_user_edit_view, name='user-edit'),
            path('users/<str:uid>/delete/', self.firestore_user_delete_view, name='user-delete'),
            path('pets/', self.firestore_pets_view, name='pets'),
            path('pets/<str:owner_uid>/<str:pet_id>/edit/', self.firestore_pet_edit_view, name='pet-edit'),
            path('pets/<str:owner_uid>/<str:pet_id>/delete/', self.firestore_pet_delete_view, name='pet-delete'),
            path('logs/', self.firestore_logs_view, name='logs'),
            path('logs/<str:owner_uid>/<str:log_id>/delete/', self.firestore_log_delete_view, name='log-delete'),
            # Legacy redirects to preserve existing links
            path('firestore-dashboard/', lambda request: redirect('/admin-panel/dashboard/')),
            path('firestore-users/', lambda request: redirect('/admin-panel/users/')),
            path('firestore-pets/', lambda request: redirect('/admin-panel/pets/')),
            path('firestore-logs/', lambda request: redirect('/admin-panel/logs/')),
        ]
        return custom_urls + urls

    @method_decorator(staff_member_required)
    def firestore_users_view(self, request):
        db = get_firestore()
        users = []
        name_q = (request.GET.get('name') or '').strip().lower()
        email_q = (request.GET.get('email') or '').strip().lower()
        number_q = (request.GET.get('number') or '').strip().lower()
        pverified_q = (request.GET.get('phoneVerified') or '').strip().lower()
        for d in db.collection('users').stream():
            u = d.to_dict() or {}
            u['uid'] = d.id
            if name_q and name_q not in (u.get('name') or '').lower():
                continue
            if email_q and email_q not in (u.get('email') or '').lower():
                continue
            if number_q and number_q not in (u.get('number') or '').lower():
                continue
            if pverified_q in ('yes','true','1') and not u.get('phoneVerified'):
                continue
            if pverified_q in ('no','false','0') and u.get('phoneVerified'):
                continue
            users.append(u)
        context = {**self.each_context(request), 'users': users}
        return TemplateResponse(request, 'admin/firestore_users.html', context)

    @method_decorator(staff_member_required)
    def firestore_user_edit_view(self, request, uid: str):
        db = get_firestore()
        ref = db.collection('users').document(uid)
        snap = ref.get()
        if not snap.exists:
            return redirect('../')
        data = snap.to_dict() or {}
        if request.method == 'POST':
            name = request.POST.get('name') or ''
            number = request.POST.get('number') or ''
            phone_verified = request.POST.get('phoneVerified') == 'on'
            ref.update({'name': name, 'number': number, 'phoneVerified': phone_verified})
            return redirect('/dashboard/users/')
        context = {**self.each_context(request), 'user_doc': {**data, 'uid': uid}}
        return TemplateResponse(request, 'admin/firestore_user_edit.html', context)

    @method_decorator(staff_member_required)
    @require_http_methods(["POST"]) 
    def firestore_user_delete_view(self, request, uid: str):
        db = get_firestore()
        # Delete subcollections (pets, emotion_logs) then the user document
        for sub in ('pets', 'emotion_logs'):
            for d in db.collection('users').document(uid).collection(sub).stream():
                d.reference.delete()
        db.collection('users').document(uid).delete()
        return redirect('/dashboard/users/')

    @method_decorator(staff_member_required)
    def firestore_pets_view(self, request):
        db = get_firestore()
        pets = []
        name_q = (request.GET.get('name') or '').strip().lower()
        gender_q = (request.GET.get('gender') or '').strip().lower()
        species_q = (request.GET.get('species') or '').strip().lower()
        breed_q = (request.GET.get('breed') or '').strip().lower()
        for d in db.collection_group('pets').limit(500).stream():
            p = d.to_dict() or {}
            p['id'] = d.id
            # owner uid is document path: users/{uid}/pets/{id}
            parts = d.reference.path.split('/')
            owner_uid = parts[1] if len(parts) >= 2 else ''
            p['ownerUid'] = owner_uid
            # Apply simple in-memory filters (Firestore composite for production)
            if name_q and name_q not in (p.get('name') or '').lower():
                continue
            if gender_q and gender_q != (p.get('gender') or '').lower():
                continue
            if species_q and species_q != (p.get('species') or '').lower():
                continue
            if breed_q and breed_q not in (p.get('breed') or '').lower():
                continue
            pets.append(p)
        context = {**self.each_context(request), 'pets': pets}
        return TemplateResponse(request, 'admin/firestore_pets.html', context)

    @method_decorator(staff_member_required)
    def firestore_pet_edit_view(self, request, owner_uid: str, pet_id: str):
        db = get_firestore()
        ref = db.collection('users').document(owner_uid).collection('pets').document(pet_id)
        snap = ref.get()
        if not snap.exists:
            return redirect('/dashboard/pets/')
        data = snap.to_dict() or {}
        if request.method == 'POST':
            name = request.POST.get('name') or ''
            gender = request.POST.get('gender') or ''
            species = request.POST.get('species') or ''
            breed = request.POST.get('breed') or ''
            dob = request.POST.get('dateOfBirth') or ''
            update = {'name': name, 'gender': gender, 'species': species, 'breed': breed}
            if dob:
                update['dateOfBirth'] = dob
            ref.update(update)
            return redirect('/dashboard/firestore-pets/')
        context = {**self.each_context(request), 'pet': {**data, 'id': pet_id, 'ownerUid': owner_uid}}
        return TemplateResponse(request, 'admin/firestore_pet_edit.html', context)

    @method_decorator(staff_member_required)
    @require_http_methods(["POST"]) 
    def firestore_pet_delete_view(self, request, owner_uid: str, pet_id: str):
        db = get_firestore()
        db.collection('users').document(owner_uid).collection('pets').document(pet_id).delete()
        return redirect('/dashboard/pets/')

    @method_decorator(staff_member_required)
    def firestore_logs_view(self, request):
        db = get_firestore()
        logs = []
        pet_id_q = (request.GET.get('petId') or '').strip()
        emotion_q = (request.GET.get('emotion') or '').strip().lower()
        min_conf_q = request.GET.get('minConfidence')
        min_conf = float(min_conf_q) if min_conf_q else 0.0
        
        try:
            docs_list = list(db.collection_group('emotion_logs').order_by('timestamp', direction='DESCENDING').limit(200).stream())
        except Exception:
            # Fallback without ordering if index missing
            docs_list = list(db.collection_group('emotion_logs').limit(200).stream())
        for d in docs_list:
            l = d.to_dict() or {}
            l['id'] = d.id
            parts = d.reference.path.split('/')
            owner_uid = parts[1] if len(parts) >= 2 else ''
            l['ownerUid'] = owner_uid
            
            # Apply filters
            if pet_id_q and pet_id_q not in (l.get('petId') or ''):
                continue
            if emotion_q and emotion_q != (l.get('emotion') or '').lower():
                continue
            if min_conf > 0 and (l.get('confidence') or 0) < min_conf:
                continue
            logs.append(l)
        # In case we fell back to unordered results, sort locally by timestamp desc
        try:
            logs.sort(key=lambda x: x.get('timestamp') or 0, reverse=True)
        except Exception:
            pass
        context = {**self.each_context(request), 'logs': logs}
        return TemplateResponse(request, 'admin/firestore_logs.html', context)

    @method_decorator(staff_member_required)
    @require_http_methods(["POST"]) 
    def firestore_log_delete_view(self, request, owner_uid: str, log_id: str):
        db = get_firestore()
        db.collection('users').document(owner_uid).collection('emotion_logs').document(log_id).delete()
        return redirect('/dashboard/logs/')


custom_admin_site = FirestoreDashboardAdmin(name='custom_admin')


