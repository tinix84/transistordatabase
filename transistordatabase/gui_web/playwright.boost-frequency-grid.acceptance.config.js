import { defineConfig, devices } from '@playwright/test'
const root=process.env.TDB_BOOST_GRID_OUTPUT
if(!root)throw new Error('Set a fresh unique Boost-grid output directory')
export default defineConfig({
 testDir:'./src/tests/e2e',testMatch:'boost-frequency-grid-acceptance.spec.js',
 outputDir:`${root}/test-results`,fullyParallel:false,workers:1,retries:0,
 timeout:60_000,expect:{timeout:15_000},
 reporter:[['line'],['json',{outputFile:`${root}/results.json`}]],
 use:{baseURL:'http://127.0.0.1:8004',trace:'off',screenshot:'only-on-failure'},
 projects:[{name:'desktop',use:{...devices['Desktop Chrome']}},{name:'mobile-390',use:{browserName:'chromium',viewport:{width:390,height:844},isMobile:true,hasTouch:true}}]
})
