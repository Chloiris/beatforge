import { expect, test } from '@playwright/test';

test('workspace to precise edit and export flow', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: '歌曲工作区' })).toBeVisible();
  for (const title of ['霓虹脉冲', '钢铁断层', '玻璃潮汐']) {
    await expect(page.locator(`article.project-card-featured[data-project-title="${title}"]`)).toBeVisible();
  }

  await page.getByRole('link', { name: '打开 霓虹脉冲' }).click();
  await expect(page.getByRole('dialog', { name: '先告诉卡点工坊：你想卡什么？' })).toBeVisible();
  await page.getByRole('button', { name: '直接编辑已有点' }).click();
    await expect(page.getByTestId('timeline-canvas')).toBeVisible();
    await expect(page.getByText(/SAMPLES/)).toBeVisible();
    await expect(page.getByRole('button', { name: 'Vocals' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Melody' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Drums' })).toBeVisible();
  const projectId = new URL(page.url()).pathname.split('/').pop()!;
  const detail = await page.request.get(`/api/projects/${projectId}`);
  expect(detail.ok()).toBeTruthy();
  const project = await detail.json();
  const originalBpm = project.track.tempoMap[0].bpm;
  try {
    const bpmInput = page.getByRole('spinbutton', { name: 'BPM', exact: true });
    await expect(bpmInput).toHaveValue(String(originalBpm));

    const audio = page.locator('audio');
    await page.getByRole('button', { name: '播放' }).click();
    await expect.poll(() => audio.evaluate((element: HTMLAudioElement) => !element.paused)).toBe(true);
    await page.getByRole('button', { name: '暂停' }).click();

    const changedBpm = originalBpm + 2;
    await bpmInput.fill(String(changedBpm));
    await expect(bpmInput).toHaveValue(String(changedBpm));

    const candidate = project.track.hitPoints.find((point: { sample: number }) => point.sample / project.track.sampleCount > 0.15 && point.sample / project.track.sampleCount < 0.8);
    expect(candidate).toBeTruthy();
    const canvas = page.getByTestId('timeline-canvas');
    const box = await canvas.boundingBox();
    expect(box).toBeTruthy();
    const x = box!.x + candidate.sample / project.track.sampleCount * box!.width;
    const y = box!.y + 150;
    await page.getByRole('switch', { name: '吸附到网格' }).click();
    await page.mouse.click(x, y);
    const sampleInput = page.getByLabel('击打点 acoustic sample');
    await expect(sampleInput).toBeVisible();
    const selectedSampleBeforeDrag = await sampleInput.inputValue();
    await page.mouse.move(x, y);
    await page.mouse.down();
    await page.mouse.move(x + 18, y, { steps: 5 });
    await page.mouse.up();
    await expect(sampleInput).not.toHaveValue(selectedSampleBeforeDrag);
    await page.getByRole('button', { name: '撤销' }).click();
    await expect(sampleInput).toHaveValue(selectedSampleBeforeDrag);
    await page.getByRole('button', { name: '撤销' }).click();
    await expect(bpmInput).toHaveValue(String(originalBpm));

    await page.locator('.export-menu summary').click();
    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('link', { name: '导出 JSON' }).click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/\.json$/);
  } finally {
    // A real reanalysis leaves the seeded demo pristine even when this test fails or retries.
    const restore = await page.request.post(`/api/tracks/${project.track.id}/analyze`, { data: { mode: 'balanced', sensitivity: 0.5 } });
    if (restore.ok()) {
      const { jobId } = await restore.json();
      for (let attempt = 0; attempt < 120; attempt += 1) {
        const response = await page.request.get(`/api/analysis-jobs/${jobId}`);
        const job = await response.json();
        if (job.status === 'completed' || job.status === 'failed') break;
        await page.waitForTimeout(250);
      }
    }
  }
});
