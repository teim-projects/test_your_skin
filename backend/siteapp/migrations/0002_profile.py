from django.db import migrations, models
import siteapp.models
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        ('siteapp', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Profile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('full_name', models.CharField(blank=True, max_length=255)),
                ('age', models.PositiveIntegerField(blank=True, null=True)),
                ('gender', models.CharField(blank=True, max_length=32)),
                ('city', models.CharField(blank=True, max_length=128)),
                ('mobile', models.CharField(blank=True, max_length=20)),
                ('location', models.CharField(blank=True, max_length=255)),
                ('bio', models.TextField(blank=True)),
                ('avatar', models.ImageField(blank=True, null=True, upload_to=siteapp.models.upload_to)),
                ('user', models.OneToOneField(on_delete=models.deletion.CASCADE, related_name='profile', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]


